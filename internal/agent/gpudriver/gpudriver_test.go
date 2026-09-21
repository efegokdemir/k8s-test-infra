// Copyright 2026 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

package gpudriver

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/NVIDIA/k8s-test-infra/internal/fsutil"

	"github.com/stretchr/testify/require"

	"github.com/NVIDIA/k8s-test-infra/internal/agent"
	"github.com/NVIDIA/k8s-test-infra/internal/agent/host"
)

// testState returns a minimal State for gpudriver tests with a real engine YAML.
func testState(t *testing.T) *agent.State {
	t.Helper()
	cfgPath := filepath.Join("..", "..", "..", "pkg", "gpu", "mocknvml", "configs", "mock-nvml-config-a100.yaml")
	data, err := os.ReadFile(cfgPath)
	require.NoError(t, err)
	return &agent.State{
		Software:  agent.SoftwareVersions{DriverVersion: "550.163.01"},
		Devices:   []agent.DeviceSpec{{Index: 0, MinorNumber: 0}},
		ConfigRaw: data,
	}
}

func testHost(t *testing.T) *host.Host {
	t.Helper()
	return host.New(t.TempDir())
}

// skipUnlessRootLinux skips the test on any platform where mknod requires root.
func skipUnlessRootLinux(t *testing.T) {
	t.Helper()
	if runtime.GOOS != "linux" || os.Getuid() != 0 {
		t.Skip("requires root on Linux (mknod)")
	}
}

// skipUnlessNVMLLib skips the test when the NVML shim .so is not installed.
func skipUnlessNVMLLib(t *testing.T) {
	t.Helper()
	matches, _ := filepath.Glob("/usr/local/lib/libnvidia-ml.so.*.*.*")
	if len(matches) == 0 {
		t.Skip("libnvidia-ml.so not installed")
	}
}

// ─── individual surface tests ────────────────────────────────────────────────

func TestWriteProcFS_WritesVersionAndParams(t *testing.T) {
	h := testHost(t)
	state := testState(t)

	require.NoError(t, writeProcFS(t.Context(), h, state))

	versionPath := h.RootPath("driver/proc/driver/nvidia/version")
	content, err := os.ReadFile(versionPath)
	require.NoError(t, err)
	require.Contains(t, string(content), state.Software.DriverVersion,
		"version file must contain DriverVersion")

	paramsPath := h.RootPath("driver/proc/driver/nvidia/params")
	_, err = os.Stat(paramsPath)
	require.NoError(t, err, "params file must exist")
}

func TestWriteProcFS_Idempotent(t *testing.T) {
	h := testHost(t)
	state := testState(t)
	ctx := t.Context()

	require.NoError(t, writeProcFS(ctx, h, state))
	require.NoError(t, writeProcFS(ctx, h, state), "second call must not error")
}

func TestWriteEngineConfig_WritesBothLocations(t *testing.T) {
	h := testHost(t)
	state := testState(t)

	require.NoError(t, writeEngineConfig(t.Context(), h, state))

	for _, rel := range []string{"config/config.yaml", "driver/config/config.yaml"} {
		_, err := os.Stat(h.RootPath(rel))
		require.NoError(t, err, "%s must exist", rel)
	}
}

func TestWriteEngineConfig_EmptyConfigRawErrors(t *testing.T) {
	h := testHost(t)
	state := &agent.State{}

	err := writeEngineConfig(t.Context(), h, state)
	require.Error(t, err)
}

// GFD reads its machine type from a file, so the mock has to serve one: under
// kind the DMI path it defaults to is either absent or owned by the node image.
func TestWriteMachineType_ServesTheProductName(t *testing.T) {
	h := testHost(t)
	state := testState(t)
	state.Devices[0].Name = "NVIDIA GB300 NVL"

	require.NoError(t, writeMachineType(t.Context(), h, state))

	data, err := os.ReadFile(filepath.Join(h.Root, machineTypeRel))
	require.NoError(t, err)
	require.Equal(t, "NVIDIA GB300 NVL\n", string(data),
		"trailing newline mirrors how the kernel renders product_name")
}

// A missing file leaves GFD on its own default, which is the better failure:
// an empty one would label the node with the empty string.
func TestWriteMachineType_NoFileWithoutAProductName(t *testing.T) {
	h := testHost(t)
	state := testState(t)
	state.Devices[0].Name = ""

	require.NoError(t, writeMachineType(t.Context(), h, state))

	_, err := os.Stat(filepath.Join(h.Root, machineTypeRel))
	require.True(t, os.IsNotExist(err), "no product name means no file")
}

func TestWriteMachineType_NoFileWithoutDevices(t *testing.T) {
	h := testHost(t)
	state := testState(t)
	state.Devices = nil

	require.NoError(t, writeMachineType(t.Context(), h, state))

	_, err := os.Stat(filepath.Join(h.Root, machineTypeRel))
	require.True(t, os.IsNotExist(err))
}

// The file is served through the same mount as config.yaml, so it has to sit
// beside it — a path outside driver/config would never reach a container.
func TestWriteMachineType_LandsInTheServedConfigDir(t *testing.T) {
	require.Equal(t, "driver/config", filepath.Dir(machineTypeRel))
}

func TestStageNvidiaSMI_WritesSMIScript(t *testing.T) {
	h := testHost(t)
	state := testState(t)

	require.NoError(t, stageNvidiaSMI(t.Context(), h, state))

	script := h.RootPath("driver/usr/bin/nvidia-smi.sh")
	content, err := os.ReadFile(script)
	require.NoError(t, err)
	require.Contains(t, string(content), state.Software.DriverVersion)

	// Whether nvidia-smi is the ELF or a symlink, it must exist.
	_, err = os.Lstat(h.RootPath("driver/usr/bin/nvidia-smi"))
	require.NoError(t, err, "nvidia-smi must exist (ELF or symlink)")
}

func TestStageCUDAShim_NopWhenNoLib(t *testing.T) {
	matches, _ := filepath.Glob("/usr/local/lib/libcuda.so.*.*.*")
	if len(matches) > 0 {
		t.Skip("libcuda.so is present; this test covers the no-lib path")
	}
	h := testHost(t)
	state := testState(t)

	require.NoError(t, stageCUDAShim(t.Context(), h, state),
		"stageCUDAShim must not error when libcuda.so is absent")
}

func TestStageNVMLShim_CopiesLibAndCreatesLinks(t *testing.T) {
	skipUnlessNVMLLib(t)

	h := testHost(t)
	state := testState(t)

	require.NoError(t, stageNVMLShim(t.Context(), h, state))

	lib64 := h.RootPath("driver/usr/lib64")
	versioned := "libnvidia-ml.so." + state.Software.DriverVersion
	for _, name := range []string{versioned, "libnvidia-ml.so.1", "libnvidia-ml.so"} {
		_, err := os.Lstat(filepath.Join(lib64, name))
		require.NoError(t, err, "%s must exist", name)
	}
}

func TestCharDevsForDevices_UsesConfiguredMinorNumbers(t *testing.T) {
	t.Parallel()

	devices := []agent.DeviceSpec{
		{Index: 0, MinorNumber: 3},
		{Index: 1, MinorNumber: 0},
	}
	require.Equal(t, []charDev{
		{"nvidia3", 195, 3},
		{"nvidia0", 195, 0},
		{"nvidiactl", 195, 255},
		{"nvidia-uvm", 510, 0},
		{"nvidia-uvm-tools", 510, 1},
	}, charDevsForDevices(devices))
}

func TestStageCharDevs_CreatesDeviceNodes(t *testing.T) {
	skipUnlessRootLinux(t)

	h := testHost(t)
	state := testState(t)

	require.NoError(t, stageCharDevs(t.Context(), h, state))

	devRoot := h.RootPath("driver/dev")
	for _, name := range []string{"nvidia0", "nvidiactl", "nvidia-uvm", "nvidia-uvm-tools"} {
		_, err := os.Stat(filepath.Join(devRoot, name))
		require.NoError(t, err, "%s chardev must exist", name)
	}
}

// The node name and the minor it is created with both come from the driver's
// numbering, not from the NVML index, so a device whose two differ gets one
// node under its minor rather than a stray node under its index.
func TestStageCharDevs_UsesConfiguredMinorNumber(t *testing.T) {
	skipUnlessRootLinux(t)

	h := testHost(t)
	state := testState(t)
	state.Devices = []agent.DeviceSpec{{Index: 1, MinorNumber: 3}}

	require.NoError(t, stageCharDevs(t.Context(), h, state))

	devRoot := filepath.Join(h.Root, "driver/dev")
	require.FileExists(t, filepath.Join(devRoot, "nvidia3"))
	require.NoFileExists(t, filepath.Join(devRoot, "nvidia1"))
}

// ─── Apply / Revoke ──────────────────────────────────────────────────────────

func TestApply_CreatesSymlink(t *testing.T) {
	h := testHost(t)
	sim := New(h)

	require.NoError(t, sim.Apply(t.Context(), testState(t)))
	require.True(t, sim.Ready())

	link := h.RunPath("nvidia/driver")
	target, err := os.Readlink(link)
	require.NoError(t, err)
	require.Equal(t, "/var/lib/nvml-mock/driver", target)
}

func TestRevoke_RemovesSymlink(t *testing.T) {
	h := testHost(t)
	sim := New(h)

	require.NoError(t, sim.Apply(t.Context(), testState(t)))
	require.NoError(t, sim.Revoke(t.Context()))

	link := h.RunPath("nvidia/driver")
	_, err := os.Lstat(link)
	require.ErrorIs(t, err, os.ErrNotExist)
}

func TestRevoke_IdempotentWhenLinkAbsent(t *testing.T) {
	sim := New(testHost(t))

	require.NoError(t, sim.Revoke(t.Context()), "Revoke on absent symlink must not error")
}

// TestRevoke_LeavesForeignPaths covers what Revoke must not delete: /run/nvidia
// is shared with the GPU Operator, so only our own symlink is ours to remove.
func TestRevoke_LeavesForeignPaths(t *testing.T) {
	cases := []struct {
		name  string
		plant func(t *testing.T, link string)
	}{
		{"empty directory", func(t *testing.T, link string) {
			require.NoError(t, os.MkdirAll(link, 0o755))
		}},
		{"regular file", func(t *testing.T, link string) {
			require.NoError(t, fsutil.Write(link, []byte("driver"), 0o644))
		}},
		{"symlink to another driver root", func(t *testing.T, link string) {
			require.NoError(t, fsutil.Symlink("/opt/real-driver", link))
		}},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := testHost(t)
			link := h.RunPath(driverLinkRel)
			c.plant(t, link)

			require.NoError(t, New(h).Revoke(t.Context()))

			_, err := os.Lstat(link)
			require.NoError(t, err, "Revoke must leave a path it did not create")
		})
	}
}

// ─── Discard ─────────────────────────────────────────────────────────────────

func TestDiscard_NopWhenNotReady(t *testing.T) {
	sim := New(testHost(t))

	// ready is false by default — Discard must be a no-op.
	require.NoError(t, sim.Discard(t.Context()))
}

// ─── full Stage (Linux root + NVML lib required) ─────────────────────────────

func TestStage_WritesAllSurfaces(t *testing.T) {
	skipUnlessRootLinux(t)
	skipUnlessNVMLLib(t)

	h := testHost(t)
	sim := New(h)
	state := testState(t)

	require.NoError(t, sim.Stage(t.Context(), state))
	require.False(t, sim.Ready(), "Stage does not publish the driver symlink")
	require.NoError(t, sim.Apply(t.Context(), state))
	require.True(t, sim.Ready())

	// chardevs
	devRoot := h.RootPath("driver/dev")
	_, err := os.Stat(filepath.Join(devRoot, "nvidiactl"))
	require.NoError(t, err)

	// NVML shim
	_, err = os.Lstat(h.RootPath("driver/usr/lib64", "libnvidia-ml.so.1"))
	require.NoError(t, err)

	// nvidia-smi
	_, err = os.Lstat(h.RootPath("driver/usr/bin/nvidia-smi"))
	require.NoError(t, err)

	// procfs
	_, err = os.Stat(h.RootPath("driver/proc/driver/nvidia/version"))
	require.NoError(t, err)

	// engine config
	_, err = os.Stat(h.RootPath("config/config.yaml"))
	require.NoError(t, err)
}

func TestStage_Idempotent(t *testing.T) {
	skipUnlessRootLinux(t)
	skipUnlessNVMLLib(t)

	h := testHost(t)
	sim := New(h)
	state := testState(t)

	require.NoError(t, sim.Stage(t.Context(), state))
	require.NoError(t, sim.Stage(t.Context(), state), "second Stage must not error")
}

// TestPruneGPUNodes_RemovesShrunkDeviceSet exercises pruneGPUNodes directly with
// plain files: pruning selects purely by name, and mknod needs root, which CI
// runners do not have.
func TestPruneGPUNodes_RemovesShrunkDeviceSet(t *testing.T) {
	devRoot := t.TempDir()
	for _, n := range []string{
		"nvidia0", "nvidia1", "nvidia2", "nvidia3",
		"nvidiactl", "nvidia-uvm", "nvidia-uvm-tools",
	} {
		require.NoError(t, os.WriteFile(filepath.Join(devRoot, n), nil, 0o600))
	}
	// The imex simulator stages the IMEX channel tree in this same directory.
	imex := filepath.Join(devRoot, "nvidia-caps-imex-channels")
	require.NoError(t, os.MkdirAll(imex, 0o755))

	// The device set shrank from four GPUs to two.
	wanted := map[string]bool{
		"nvidia0": true, "nvidia1": true,
		"nvidiactl": true, "nvidia-uvm": true, "nvidia-uvm-tools": true,
	}
	require.NoError(t, pruneGPUNodes(devRoot, wanted))

	for _, keep := range []string{"nvidia0", "nvidia1", "nvidiactl", "nvidia-uvm", "nvidia-uvm-tools"} {
		require.FileExists(t, filepath.Join(devRoot, keep))
	}
	for _, gone := range []string{"nvidia2", "nvidia3"} {
		require.NoFileExists(t, filepath.Join(devRoot, gone),
			"stale GPU node %s must be pruned", gone)
	}
	// The nvidia prefix alone must not be grounds for deletion — this tree
	// belongs to the imex simulator, not to gpudriver.
	require.DirExists(t, imex, "IMEX channel tree must survive pruning")
}

// TestStageCharDevs_PrunesShrunkDeviceSet guards the call site, not just the
// helper: stageCharDevs must prune GPU nodes a larger device set left behind.
func TestStageCharDevs_PrunesShrunkDeviceSet(t *testing.T) {
	skipUnlessRootLinux(t)

	h := testHost(t)
	devRoot := h.RootPath("driver/dev")
	require.NoError(t, os.MkdirAll(devRoot, 0o755))

	// A previous, larger device set left four GPU nodes behind.
	for i, n := range []string{"nvidia0", "nvidia1", "nvidia2", "nvidia3"} {
		require.NoError(t, fsutil.Mknod(filepath.Join(devRoot, n), 195, uint32(i)))
	}

	state := &agent.State{Devices: []agent.DeviceSpec{
		{Index: 0, MinorNumber: 0},
		{Index: 1, MinorNumber: 1},
	}}
	require.NoError(t, stageCharDevs(t.Context(), h, state))

	require.FileExists(t, filepath.Join(devRoot, "nvidia0"))
	require.FileExists(t, filepath.Join(devRoot, "nvidia1"))
	require.NoFileExists(t, filepath.Join(devRoot, "nvidia2"))
	require.NoFileExists(t, filepath.Join(devRoot, "nvidia3"))
}

// ─── Apply / Revoke: leave the node as you found it ──────────────────────────
//
// /run/nvidia/driver is shared with the GPU Operator's driver container. The
// tests below pin the whole contract: Apply may displace a foreign symlink but
// has to hand the node back unchanged, and it must refuse outright to replace a
// directory or a file it cannot put back.

// A foreign symlink is the GPU Operator's own driver root. Apply is allowed to
// point the path at us, but the node has to come back as it was.
func TestApplyRevoke_RestoresDisplacedForeignSymlink(t *testing.T) {
	h := testHost(t)
	sim := New(h)
	link := h.RunPath(driverLinkRel)
	require.NoError(t, fsutil.Symlink("/opt/real-driver", link))

	require.NoError(t, sim.Apply(t.Context(), testState(t)))

	published, err := os.Readlink(link)
	require.NoError(t, err)
	require.Equal(t, driverLinkTarget, published, "Apply still publishes our driver root")

	require.NoError(t, sim.Revoke(t.Context()))

	restored, err := os.Readlink(link)
	require.NoError(t, err, "the displaced driver root must be put back, not deleted")
	require.Equal(t, "/opt/real-driver", restored)
}

// Re-applying finds our own symlink. The displaced target was recorded by the
// first Apply and must survive, or the second Apply silently forgets how to put
// the node back.
func TestApplyTwice_KeepsTheDisplacedForeignTarget(t *testing.T) {
	h := testHost(t)
	sim := New(h)
	link := h.RunPath(driverLinkRel)
	require.NoError(t, fsutil.Symlink("/opt/real-driver", link))

	require.NoError(t, sim.Apply(t.Context(), testState(t)))
	require.NoError(t, sim.Apply(t.Context(), testState(t)))
	require.NoError(t, sim.Revoke(t.Context()))

	restored, err := os.Readlink(link)
	require.NoError(t, err)
	require.Equal(t, "/opt/real-driver", restored,
		"the second Apply must not drop the first Apply's displacement record")
}

// Once the path is gone there is nothing left to restore: resurrecting a driver
// root somebody else deleted would be its own kind of damage.
func TestApply_ForgetsTheDisplacedTargetOnceThePathIsGone(t *testing.T) {
	h := testHost(t)
	sim := New(h)
	link := h.RunPath(driverLinkRel)
	require.NoError(t, fsutil.Symlink("/opt/real-driver", link))

	require.NoError(t, sim.Apply(t.Context(), testState(t)))
	require.NoError(t, os.Remove(link))
	require.NoError(t, sim.Apply(t.Context(), testState(t)))
	require.NoError(t, sim.Revoke(t.Context()))

	_, err := os.Lstat(link)
	require.ErrorIs(t, err, os.ErrNotExist,
		"nothing was displaced by the second Apply, so Revoke just removes our link")
}

// A directory or a file cannot be displaced and put back, so Apply refuses.
// fsutil.Symlink would unlink an empty directory or a regular file outright,
// which is why those two cases are here and not just the populated one.
func TestApply_RefusesToReplaceAForeignDirectoryOrFile(t *testing.T) {
	cases := []struct {
		name  string
		names string
		plant func(t *testing.T, link string)
		check func(t *testing.T, link string)
	}{
		{
			name:  "populated GPU-Operator driver root",
			names: "directory",
			plant: func(t *testing.T, link string) {
				lib := filepath.Join(link, "usr/lib64/libnvidia-ml.so.1")
				require.NoError(t, os.MkdirAll(filepath.Dir(lib), 0o755))
				require.NoError(t, os.WriteFile(lib, []byte("real driver"), 0o644))
			},
			check: func(t *testing.T, link string) {
				data, err := os.ReadFile(filepath.Join(link, "usr/lib64/libnvidia-ml.so.1"))
				require.NoError(t, err)
				require.Equal(t, "real driver", string(data))
			},
		},
		{
			name:  "empty directory",
			names: "directory",
			plant: func(t *testing.T, link string) {
				require.NoError(t, os.MkdirAll(link, 0o755))
			},
			check: func(t *testing.T, link string) {
				fi, err := os.Lstat(link)
				require.NoError(t, err, "an empty directory must not be unlinked either")
				require.True(t, fi.IsDir())
			},
		},
		{
			name:  "regular file",
			names: "regular file",
			plant: func(t *testing.T, link string) {
				require.NoError(t, fsutil.Write(link, []byte("driver"), 0o644))
			},
			check: func(t *testing.T, link string) {
				data, err := os.ReadFile(link)
				require.NoError(t, err)
				require.Equal(t, "driver", string(data))
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := testHost(t)
			sim := New(h)
			link := h.RunPath(driverLinkRel)
			c.plant(t, link)

			err := sim.Apply(t.Context(), testState(t))

			require.Error(t, err, "Apply must fail fast rather than destroy a foreign driver root")
			require.ErrorContains(t, err, link, "the error must name the path")
			require.ErrorContains(t, err, c.names, "the error must name what is there")
			require.False(t, sim.Ready())
			c.check(t, link)
		})
	}
}

// Publishing the symlink is the point of Apply: a probe that cannot read the
// path must not cost us the whole apply. Nothing is recorded, so Revoke will
// not invent a restore either.
func TestApply_ProbeFailureWarnsAndPublishesAnyway(t *testing.T) {
	h := testHost(t)
	sim := New(h)
	link := h.RunPath(driverLinkRel)

	restore := lstat
	lstat = func(string) (os.FileInfo, error) { return nil, errors.New("boom") }
	t.Cleanup(func() { lstat = restore })

	require.NoError(t, sim.Apply(t.Context(), testState(t)),
		"a failed ownership probe must not abort Apply")
	require.True(t, sim.Ready())

	target, err := os.Readlink(link)
	require.NoError(t, err)
	require.Equal(t, driverLinkTarget, target)
}

// The mirror of the rule above: Revoke deletes only what it positively
// identified as ours, so a probe failure leaves the path alone.
func TestRevoke_ProbeFailureLeavesThePathAlone(t *testing.T) {
	h := testHost(t)
	sim := New(h)
	link := h.RunPath(driverLinkRel)
	require.NoError(t, sim.Apply(t.Context(), testState(t)))

	restore := lstat
	lstat = func(string) (os.FileInfo, error) { return nil, errors.New("boom") }
	t.Cleanup(func() { lstat = restore })

	require.Error(t, sim.Revoke(t.Context()))

	_, err := os.Lstat(link)
	require.NoError(t, err, "Revoke must not delete a path it could not identify")
}

// Revoke never displaced this symlink, so it is not ours to restore or remove.
func TestRevoke_LeavesAForeignSymlinkItNeverDisplaced(t *testing.T) {
	h := testHost(t)
	link := h.RunPath(driverLinkRel)
	require.NoError(t, fsutil.Symlink("/opt/real-driver", link))

	require.NoError(t, New(h).Revoke(t.Context()))

	target, err := os.Readlink(link)
	require.NoError(t, err)
	require.Equal(t, "/opt/real-driver", target)
}
