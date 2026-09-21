// Copyright 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Package gpudriver implements the GPU driver footprint simulator:
// chardevs, NVML/CUDA shims, nvidia-smi, procfs entries, engine config,
// and the /run/nvidia/driver GPU-Operator compatibility symlink.
package gpudriver

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync/atomic"

	"go.uber.org/zap"
	"golang.org/x/sync/errgroup"

	"github.com/NVIDIA/k8s-test-infra/internal/fsutil"

	"github.com/NVIDIA/k8s-test-infra/internal/agent"
	"github.com/NVIDIA/k8s-test-infra/internal/agent/host"
)

const name = "gpudriver"

// The GPU-Operator compatibility symlink and the driver root it points at.
// This resolves to the node's real /run/nvidia/driver, a path the GPU Operator's
// driver container also owns.
const (
	driverLinkRel    = "nvidia/driver"
	driverLinkTarget = "/var/lib/nvml-mock/driver"
)

var (
	_ agent.Simulator = (*Simulator)(nil)
	_ agent.Applier   = (*Simulator)(nil)
)

// lstat is a seam so tests can make the driver-root ownership probe fail; there
// is no portable way to provoke a real lstat error on a path we can create.
var lstat = os.Lstat

// Simulator implements agent.Simulator and agent.Applier.
type Simulator struct {
	host  *host.Host
	ready atomic.Bool

	// displaced holds the target of a foreign driver-root symlink that Apply
	// moved aside, so Revoke can hand the node back the way it found it. A nil
	// pointer means we displaced nothing and Revoke has nothing to restore.
	// Apply and Revoke run in the same process, so this needs no on-disk state.
	displaced atomic.Pointer[string]
}

// New returns a gpudriver Simulator.
func New(h *host.Host) *Simulator { return &Simulator{host: h} }

// Name returns the simulator's stable identifier.
func (s *Simulator) Name() string { return name }

// Ready reports whether the driver footprint and its published symlink exist.
func (s *Simulator) Ready() bool { return s.ready.Load() }

// Stage materializes the GPU driver footprint under host.Root/driver/.
// All surfaces run in parallel; a failure in any one cancels the rest via gctx.
func (s *Simulator) Stage(ctx context.Context, state *agent.State) error {
	s.ready.Store(false)
	zap.L().Info("staging simulator", zap.String("simulator", name))

	g, gctx := errgroup.WithContext(ctx)
	g.Go(func() error { return stageCharDevs(gctx, s.host, state) })
	g.Go(func() error { return stageNVMLShim(gctx, s.host, state) })
	g.Go(func() error { return stageCUDAShim(gctx, s.host, state) })
	g.Go(func() error { return stageNvidiaSMI(gctx, s.host, state) })
	g.Go(func() error { return writeProcFS(gctx, s.host, state) })
	g.Go(func() error { return writeEngineConfig(gctx, s.host, state) })
	g.Go(func() error { return writeMachineType(gctx, s.host, state) })

	if err := g.Wait(); err != nil {
		return err
	}

	zap.L().Info("simulator staged", zap.String("simulator", name))
	return nil
}

// stagedPaths lists exactly the paths Stage writes, in removal order (leaves first).
// RemoveAll on the whole driver/ tree is intentionally avoided: the ib and pcibus
// simulators stage tools, libibverbs.d and preload shims there, and those must
// survive Discard.
var stagedPaths = []string{
	"driver/dev",
	"driver/usr/lib64",
	"driver/usr/bin/nvidia-smi",
	"driver/usr/bin/nvidia-smi.sh",
	"driver/proc/driver/nvidia",
	"driver/config/config.yaml",
	machineTypeRel,
	"config/config.yaml",
}

// Discard removes only the paths Stage wrote. Every path is exclusively owned
// by gpudriver, so removing absent or partially staged paths is safe.
func (s *Simulator) Discard(_ context.Context) error {
	zap.L().Info("discarding simulator", zap.String("simulator", name))

	var errs []error

	for _, rel := range stagedPaths {
		p := s.host.RootPath(rel)
		if err := os.RemoveAll(p); err != nil && !os.IsNotExist(err) {
			errs = append(errs, fmt.Errorf("remove %s: %w", p, err))
		}
	}

	return errors.Join(errs...)
}

// Apply publishes the GPU-Operator compatibility symlink at /run/nvidia/driver.
//
// The path is shared with the GPU Operator's driver container, so Apply leaves
// the node as it found it: a foreign symlink may be displaced but its target is
// recorded for Revoke to restore, and a directory or a file - which cannot be
// displaced and put back - is refused outright.
func (s *Simulator) Apply(_ context.Context, _ *agent.State) error {
	zap.L().Info("applying simulator", zap.String("simulator", name))
	s.ready.Store(false)

	driverLink := s.host.RunPath(driverLinkRel)

	root, err := probeDriverRoot(driverLink)

	switch {
	case err != nil:
		// Publishing the symlink is the point of Apply, so a probe that cannot
		// complete is a warning rather than a failure. Nothing gets recorded,
		// so Revoke will not try to restore a target we never managed to read.
		zap.L().Warn("cannot identify the existing driver root; publishing over it",
			zap.String("path", driverLink), zap.Error(err))

	case root.kind == driverRootOccupied:
		return fmt.Errorf("driver root %s is an existing %s belonging to another component: refusing to replace it",
			driverLink, root.what)

	case root.kind == driverRootForeign:
		zap.L().Warn("displacing another component's driver root; revoke restores it",
			zap.String("path", driverLink), zap.String("target", root.target))

		s.displaced.Store(&root.target)

	case root.kind == driverRootAbsent:
		// Nothing is there to hand back, and any earlier record is stale: the
		// driver root we displaced is already gone.
		s.displaced.Store(nil)

	case root.kind == driverRootOurs:
		// Re-applying over our own symlink. Whatever the first Apply displaced
		// is still displaced, so that record has to survive.
	}

	if err := fsutil.Symlink(driverLinkTarget, driverLink); err != nil {
		return err
	}

	s.ready.Store(true)

	return nil
}

// Revoke takes our /run/nvidia/driver symlink back down: it restores the driver
// root Apply displaced, or removes the symlink when Apply displaced nothing.
// Anything we cannot positively identify as ours is left untouched.
func (s *Simulator) Revoke(_ context.Context) error {
	zap.L().Info("revoking simulator", zap.String("simulator", name))
	s.ready.Store(false)

	link := s.host.RunPath(driverLinkRel)

	root, err := probeDriverRoot(link)
	if err != nil {
		// Deleting a path we failed to identify is exactly the damage this
		// simulator is supposed to avoid.
		return err
	}

	if root.kind != driverRootOurs {
		zap.L().Warn("driver root is not ours; leaving it alone",
			zap.String("path", link), zap.String("kind", root.String()))

		return nil
	}

	if displaced := s.displaced.Load(); displaced != nil {
		zap.L().Info("restoring the driver root we displaced",
			zap.String("path", link), zap.String("target", *displaced))

		if err := fsutil.Symlink(*displaced, link); err != nil {
			return err
		}

		s.displaced.Store(nil)

		return nil
	}

	return fsutil.Remove(link)
}

// driverRootKind classifies what occupies the driver-root path.
type driverRootKind int

const (
	driverRootAbsent   driverRootKind = iota // nothing is there
	driverRootOurs                           // the symlink Apply published
	driverRootForeign                        // a symlink to somebody else's driver root
	driverRootOccupied                       // a directory, a file, or anything else
)

// driverRoot is one observation of the driver-root path.
type driverRoot struct {
	kind driverRootKind
	// target is the symlink target, for driverRootForeign.
	target string
	// what names the occupant for a human, for driverRootOccupied.
	what string
}

// String renders the observation for a log field.
func (r driverRoot) String() string {
	switch r.kind {
	case driverRootAbsent:
		return "absent"
	case driverRootOurs:
		return "ours"
	case driverRootForeign:
		return "symlink to " + r.target
	case driverRootOccupied:
		return r.what
	}

	return "unknown"
}

// probeDriverRoot reports what is at link. An error means the path could not be
// identified at all, which callers must not confuse with "not ours".
func probeDriverRoot(link string) (driverRoot, error) {
	fi, err := lstat(link)
	if os.IsNotExist(err) {
		return driverRoot{kind: driverRootAbsent}, nil
	}

	if err != nil {
		return driverRoot{}, fmt.Errorf("lstat %s: %w", link, err)
	}

	if fi.Mode()&os.ModeSymlink == 0 {
		return driverRoot{kind: driverRootOccupied, what: describeMode(fi.Mode())}, nil
	}

	target, err := os.Readlink(link)
	if err != nil {
		return driverRoot{}, fmt.Errorf("readlink %s: %w", link, err)
	}

	if target == driverLinkTarget {
		return driverRoot{kind: driverRootOurs, target: target}, nil
	}

	return driverRoot{kind: driverRootForeign, target: target}, nil
}

// describeMode names a file type the way an operator reading the error would.
func describeMode(m os.FileMode) string {
	switch {
	case m.IsDir():
		return "directory"
	case m.IsRegular():
		return "regular file"
	}

	return m.Type().String()
}
