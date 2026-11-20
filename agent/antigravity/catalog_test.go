package antigravity

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/chenhg5/cc-connect/core"
)

func TestParseAgyModelCatalog_JSONFixture(t *testing.T) {
	raw := readFixture(t, "models.json")
	models, err := parseAgyModelCatalog(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 7 {
		t.Fatalf("models = %d, want 7", len(models))
	}
	if models[0].Name != "gemini-3.8-flash-low" || models[0].Desc != "Gemini 3.8 Flash (Low)" {
		t.Fatalf("first model = %+v", models[0])
	}
	if models[5].Name != "gemini-pro-orchestrator" || models[6].Name != "claude-sonnet-4-6" {
		t.Fatalf("tail = %+v %+v", models[5], models[6])
	}
	for _, stale := range []string{"gemini-2.5-pro", "gemini-2.5-flash", "gemini-3.1-pro-preview", "gemini-3-flash-preview"} {
		for _, model := range models {
			if model.Name == stale {
				t.Fatalf("catalog contains stale Gemini API id %s", stale)
			}
		}
	}
}

func TestParseAgyModelCatalog_TextFixture(t *testing.T) {
	models, err := parseAgyModelCatalog(readFixture(t, "models.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 5 || models[1].Name != "gemini-3.8-flash-medium" || models[1].Desc != "Gemini 3.8 Flash (Medium)" {
		t.Fatalf("models = %+v", models)
	}
}

func TestAvailableModels_UsesAgyCatalogNotGeminiAPI(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "ya29.test-secret")
	t.Setenv("GOOGLE_API_KEY", "ya29.test-secret")
	var calls [][]string
	agent := &Agent{
		cmd: "agy",
		run: func(_ context.Context, _ string, args ...string) (string, string, error) {
			calls = append(calls, append([]string(nil), args...))
			for _, arg := range args {
				if strings.Contains(arg, "generativelanguage.googleapis.com") || strings.Contains(arg, "http") {
					t.Fatalf("model discovery called the network: %v", args)
				}
			}
			return readFixture(t, "models.json"), "", nil
		},
		launch: &launchState{caps: cliCapabilities{ModelsCommand: true, OutputFormat: true, Model: true}},
	}

	models := agent.AvailableModels(context.Background())
	if len(models) != 7 || models[2].Name != "gemini-3.8-flash-high" {
		t.Fatalf("models = %+v", models)
	}
	if len(calls) != 1 || strings.Join(calls[0], " ") != "--output-format json models" {
		t.Fatalf("calls = %q", calls)
	}
}

func TestAvailableModels_OldCLIDoesNotInventModels(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "secret-key")
	agent := &Agent{cmd: "agy", launch: &launchState{}}
	if models := agent.AvailableModels(context.Background()); len(models) != 0 {
		t.Fatalf("models = %+v, want none", models)
	}
}

func TestResolveAgyLaunch_ModelVariantAndEffort(t *testing.T) {
	catalog, err := parseAgyModelCatalog(readFixture(t, "models.json"))
	if err != nil {
		t.Fatal(err)
	}
	caps := cliCapabilities{Model: true, Effort: true}
	got, err := resolveAgyLaunch("gemini-3.8-flash-medium", "high", caps, catalog)
	if err != nil {
		t.Fatal(err)
	}
	if got.Model != "gemini-3.8-flash-high" || got.Effort != "" {
		t.Fatalf("launch = %+v, want variant slug and no --effort", got)
	}

	_, err = resolveAgyLaunch("claude-sonnet-4-6", "high", caps, catalog)
	if !errors.Is(err, errInvalidAgyCombo) {
		t.Fatalf("fixed model err = %v, want invalid combination", err)
	}
	got, err = resolveAgyLaunch("custom-base", "high", caps, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.Model != "custom-base" || got.Effort != "high" {
		t.Fatalf("uncatalogued base launch = %+v", got)
	}

	_, err = resolveAgyLaunch("gemini-3.1-pro-high", "medium", caps, catalog)
	if !errors.Is(err, errInvalidAgyCombo) {
		t.Fatalf("err = %v, want invalid combination", err)
	}
	_, err = resolveAgyLaunch("gemini-pro-orchestrator", "high", caps, catalog)
	if !errors.Is(err, errInvalidAgyCombo) {
		t.Fatalf("orchestrator err = %v, want invalid combination", err)
	}

	old := cliCapabilities{}
	got, err = resolveAgyLaunch("gemini-3.8-flash-high", "", old, catalog)
	if err != nil || got.Model != "" || got.Effort != "" {
		t.Fatalf("old CLI launch = %+v, err = %v", got, err)
	}
}

func TestBuildArgs_PassesModelWithoutShell(t *testing.T) {
	catalog, err := parseAgyModelCatalog(readFixture(t, "models.json"))
	if err != nil {
		t.Fatal(err)
	}
	session, err := newAntigravitySession(context.Background(), "agy", nil, `F:\Antigravity`, "gemini-3.8-flash-medium", "yolo", "conv-1", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close() }()
	session.launch.mu.Lock()
	session.launch.caps = cliCapabilities{Model: true, Effort: true}
	session.launch.effort = "high"
	session.launch.catalog = catalog
	session.launch.mu.Unlock()

	resolved, err := session.resolvedLaunch()
	if err != nil {
		t.Fatal(err)
	}
	args := session.buildAntigravityArgs("conv-1", true, "yolo", "", "hello", resolved)
	if !containsSequence(args, []string{"--model", "gemini-3.8-flash-high"}) {
		t.Fatalf("args = %v, want --model variant", args)
	}
	if contains(args, "--effort") {
		t.Fatalf("args = %v, variant must not also pass --effort", args)
	}
	joined := strings.Join(args, "\x00")
	if strings.Contains(joined, `"`) || strings.Contains(strings.ToLower(joined), "cmd.exe") || strings.Contains(joined, "/c") {
		t.Fatalf("args are not a direct argv list: %v", args)
	}
	if args[len(args)-2] != "-p" || args[len(args)-1] != "hello" {
		t.Fatalf("prompt must stay last: %v", args)
	}
}

func TestReasoningPersistenceAndInvalidCombo(t *testing.T) {
	catalog, err := parseAgyModelCatalog(readFixture(t, "models.json"))
	if err != nil {
		t.Fatal(err)
	}
	agent := &Agent{
		model: "gemini-3.8-flash-medium",
		mode:  "yolo",
		cmd:   "agy",
		launch: &launchState{
			model:     "gemini-3.8-flash-medium",
			caps:      cliCapabilities{Model: true, Effort: true, ModelsCommand: true},
			catalog:   catalog,
			catalogAt: time.Now(),
		},
	}
	agent.SetReasoningEffort("high")
	if got := agent.GetModel(); got != "gemini-3.8-flash-high" {
		t.Fatalf("model = %q, want high variant", got)
	}
	if got := agent.GetReasoningEffort(); got != "high" {
		t.Fatalf("effort = %q", got)
	}
	model, persist := agent.ReasoningModel()
	if !persist || model != "gemini-3.8-flash-high" {
		t.Fatalf("persist model = %q, %v", model, persist)
	}
	if _, persist = agent.ReasoningModel(); persist {
		t.Fatal("persist flag should be consumed")
	}

	agent.SetReasoningEffort("max")
	if got := agent.GetModel(); got != "gemini-3.8-flash-high" {
		t.Fatalf("invalid effort changed model to %q", got)
	}
	opts := agent.WorkspaceAgentOptions()
	if opts["model"] != "gemini-3.8-flash-high" || opts["reasoning_effort"] != "high" || opts["mode"] != "yolo" {
		t.Fatalf("workspace options = %#v", opts)
	}
}

func TestProbeAgyCLI_VersionAndHelp(t *testing.T) {
	help := "Usage: agy.exe [flags] [command]\nFlags:\n  --model string\n  --effort string\n  --output-format string\nCommands:\n  models\n"
	caps := probeAgyCLI(context.Background(), func(_ context.Context, _ string, args ...string) (string, string, error) {
		switch strings.Join(args, " ") {
		case "--version":
			return "1.5.12", "", nil
		case "--help":
			return help, "", nil
		default:
			return "", "", errors.New("unexpected")
		}
	}, "agy")
	if !caps.Model || !caps.Effort || !caps.OutputFormat || !caps.ModelsCommand || !caps.UsagePrint || caps.Version != "1.5.12" {
		t.Fatalf("caps = %+v", caps)
	}

	old := probeAgyCLI(context.Background(), func(_ context.Context, _ string, args ...string) (string, string, error) {
		if args[0] == "--version" {
			return "1.1.10", "", nil
		}
		return help, "", nil
	}, "agy")
	if old.UsagePrint || !old.Model || !old.ModelsCommand {
		t.Fatalf("old caps = %+v", old)
	}

	noSubcommand := probeAgyCLI(context.Background(), func(_ context.Context, _ string, args ...string) (string, string, error) {
		switch args[0] {
		case "--version":
			return "1.5.12", "", nil
		default:
			return "", "Usage: agy.exe\n  --model string\n  --effort string\n  --output-format string\n", errors.New("exit status 1")
		}
	}, "agy")
	if !noSubcommand.Model || !noSubcommand.ModelsCommand || !noSubcommand.UsagePrint {
		t.Fatalf("help-without-models-word caps = %+v", noSubcommand)
	}
	if !noSubcommand.canListModels() {
		t.Fatal("1.5.12 should list models even when --help omits the subcommand")
	}
	if (cliCapabilities{}).canListModels() {
		t.Fatal("unprobed empty caps must not shell out")
	}
}

func TestAvailableModels_SlowCLIStillReturnsCatalog(t *testing.T) {
	agent := &Agent{
		cmd: "agy",
		launch: &launchState{caps: cliCapabilities{
			Model: true, ModelsCommand: true, OutputFormat: true,
		}},
		run: func(ctx context.Context, _ string, _ ...string) (string, string, error) {
			select {
			case <-ctx.Done():
				return "", "", ctx.Err()
			case <-time.After(1200 * time.Millisecond):
				return readFixture(t, "models.json"), "", nil
			}
		},
	}
	short, cancelShort := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancelShort()
	if models := agent.AvailableModels(short); len(models) != 0 {
		t.Fatalf("short deadline returned %d models", len(models))
	}
	long, cancelLong := context.WithTimeout(context.Background(), 4*time.Second)
	defer cancelLong()
	if models := agent.AvailableModels(long); len(models) != 7 {
		t.Fatalf("catalog within the card budget = %d, want 7", len(models))
	}
}

func TestParseAgyUsage_RemainingQuotaFixture(t *testing.T) {
	now := time.Date(2026, 11, 7, 11, 1, 57, 0, time.UTC)
	report, err := parseAgyUsageReport(readFixture(t, "usage.json"), now)
	if err != nil {
		t.Fatal(err)
	}
	if report.Provider != "antigravity" || len(report.Buckets) != 2 {
		t.Fatalf("report = %+v", report)
	}
	weekly := report.Buckets[0].Windows[0]
	if weekly.WindowSeconds != 604800 || weekly.UsedPercent != 4 {
		t.Fatalf("weekly = %+v, want 96%% remaining", weekly)
	}
	fiveHour := report.Buckets[0].Windows[1]
	if fiveHour.WindowSeconds != 18000 || fiveHour.UsedPercent != 5 {
		t.Fatalf("5h = %+v, want 95%% remaining", fiveHour)
	}
	if fiveHour.ResetAfterSeconds != 3*3600+58*60+12 {
		t.Fatalf("reset = %d", fiveHour.ResetAfterSeconds)
	}
}

func TestParseAgyUsage_RejectsConsumedTokens(t *testing.T) {
	_, err := parseAgyUsageReport(readFixture(t, "usage-tokens-only.json"), time.Now())
	if !errors.Is(err, core.ErrUsageUnsupported) {
		t.Fatalf("err = %v, want unsupported", err)
	}
}

func TestGetUsage_RoutesUnsupportedAndSanitizesErrors(t *testing.T) {
	t.Run("unsupported does not run a prompt", func(t *testing.T) {
		called := false
		agent := &Agent{
			cmd:    "agy",
			launch: &launchState{caps: cliCapabilities{Model: true}},
			run: func(context.Context, string, ...string) (string, string, error) {
				called = true
				return "", "", nil
			},
		}
		_, err := agent.GetUsage(context.Background())
		if !errors.Is(err, core.ErrUsageUnsupported) || called {
			t.Fatalf("err = %v, called = %v", err, called)
		}
	})

	t.Run("quota command", func(t *testing.T) {
		agent := &Agent{
			cmd:    "agy",
			launch: &launchState{caps: cliCapabilities{UsagePrint: true, OutputFormat: true}},
			run: func(_ context.Context, _ string, args ...string) (string, string, error) {
				if strings.Join(args, " ") != "--output-format json -p /usage" {
					t.Fatalf("args = %v", args)
				}
				return readFixture(t, "usage.json"), "", nil
			},
		}
		report, err := agent.GetUsage(context.Background())
		if err != nil || len(report.Buckets) != 2 {
			t.Fatalf("report = %+v, err = %v", report, err)
		}
	})

	t.Run("token answer is unsupported", func(t *testing.T) {
		agent := &Agent{
			cmd:    "agy",
			launch: &launchState{caps: cliCapabilities{UsagePrint: true}},
			run: func(context.Context, string, ...string) (string, string, error) {
				return readFixture(t, "usage-tokens-only.json"), "", nil
			},
		}
		_, err := agent.GetUsage(context.Background())
		if !errors.Is(err, core.ErrUsageUnsupported) {
			t.Fatalf("err = %v", err)
		}
	})

	t.Run("error redacts secrets", func(t *testing.T) {
		agent := &Agent{
			cmd:    "agy",
			launch: &launchState{caps: cliCapabilities{UsagePrint: true}},
			run: func(context.Context, string, ...string) (string, string, error) {
				return "", "auth failed bearer ya29.super-secret-token api_key=abcd", errors.New("exit status 1")
			},
		}
		_, err := agent.GetUsage(context.Background())
		if err == nil || strings.Contains(err.Error(), "super-secret") || strings.Contains(err.Error(), "abcd") {
			t.Fatalf("err = %v", err)
		}
		if !strings.Contains(err.Error(), "[redacted]") {
			t.Fatalf("err = %v, want redacted", err)
		}
	})

	t.Run("short deadline does not spawn", func(t *testing.T) {
		called := false
		agent := &Agent{
			cmd:    "agy",
			launch: &launchState{caps: cliCapabilities{UsagePrint: true}},
			run: func(context.Context, string, ...string) (string, string, error) {
				called = true
				return "", "", nil
			},
		}
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		_, err := agent.GetUsage(ctx)
		if err == nil || called || strings.Contains(err.Error(), "quota") && strings.Contains(err.Error(), "%") {
			t.Fatalf("err = %v, called = %v", err, called)
		}
	})
}

func TestSend_InvokesSelectedModel(t *testing.T) {
	if os.Getenv("GO_WANT_ANTIGRAVITY_MODEL_HELPER") == "1" {
		path := os.Getenv("CC_ANTIGRAVITY_ARGS_FILE")
		if path == "" {
			os.Exit(2)
		}
		if err := os.WriteFile(path, []byte(strings.Join(os.Args, "\x00")), 0o600); err != nil {
			os.Exit(2)
		}
		_, _ = os.Stdout.WriteString("{\"event\":\"result\",\"result\":{\"status\":\"SUCCESS\",\"response\":\"ok\\n\"}}\n")
		os.Exit(0)
	}

	argsPath := filepath.Join(t.TempDir(), "args")
	session, err := newAntigravitySession(
		context.Background(),
		os.Args[0],
		[]string{"-test.run=TestSend_InvokesSelectedModel", "--"},
		t.TempDir(),
		"gemini-3.8-flash-medium",
		"yolo",
		"",
		[]string{"GO_WANT_ANTIGRAVITY_MODEL_HELPER=1", "CC_ANTIGRAVITY_ARGS_FILE=" + argsPath},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close() }()
	session.launch.mu.Lock()
	session.launch.caps.Model = true
	session.launch.mu.Unlock()

	if err := session.Send("ping", "", nil, nil); err != nil {
		t.Fatal(err)
	}
	select {
	case event := <-session.Events():
		if event.Type == core.EventError {
			t.Fatal(event.Error)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("timeout")
	}
	data, err := os.ReadFile(argsPath)
	if err != nil {
		t.Fatal(err)
	}
	args := strings.Split(string(data), "\x00")
	if !containsSequence(args, []string{"--model", "gemini-3.8-flash-medium"}) {
		t.Fatalf("argv = %q", args)
	}
}

func readFixture(t *testing.T, name string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func containsSequence(args, want []string) bool {
	for i := 0; i+len(want) <= len(args); i++ {
		match := true
		for j := range want {
			if args[i+j] != want[j] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}
