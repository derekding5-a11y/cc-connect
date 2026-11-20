//go:build !windows

package antigravity

import "os/exec"

func hideConsole(cmd *exec.Cmd) {}
