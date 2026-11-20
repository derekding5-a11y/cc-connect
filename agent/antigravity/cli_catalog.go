package antigravity

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/chenhg5/cc-connect/core"
)

const (
	agyProbeTimeout   = 5 * time.Second
	agyUsageMinBudget = 5 * time.Second
	agyCatalogTTL     = 30 * time.Second
)

// agyEffortOrder is the documented /effort and --effort vocabulary.
// Availability for a model is the subset that agy models actually lists.
var agyEffortOrder = []string{"low", "medium", "high", "xhigh", "max"}

// errInvalidAgyCombo is returned when the selected model and reasoning level
// cannot be expressed with the documented --model / --effort flags.
var errInvalidAgyCombo = errors.New("invalid agy model/effort combination / 模型与推理强度组合无效")

type cliCapabilities struct {
	Model         bool
	Effort        bool
	OutputFormat  bool
	ModelsCommand bool
	UsagePrint    bool
	Probed        bool
	Version       string
}

type launchState struct {
	mu           sync.RWMutex
	model        string
	effort       string
	caps         cliCapabilities
	catalog      []core.ModelOption
	catalogAt    time.Time
	persistModel bool
}

func (s *launchState) snapshot() (model, effort string, caps cliCapabilities, catalog []core.ModelOption) {
	if s == nil {
		return "", "", cliCapabilities{}, nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.model, s.effort, s.caps, append([]core.ModelOption(nil), s.catalog...)
}

type cliRunner func(ctx context.Context, name string, args ...string) (stdout, stderr string, err error)

func execCLI(ctx context.Context, name string, args ...string) (string, string, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	hideConsole(cmd)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	return stdout.String(), stderr.String(), err
}

func (a *Agent) runCLI(ctx context.Context, args ...string) (string, string, error) {
	run := a.run
	name := a.cmd
	if name == "" {
		name = "agy"
	}
	if run == nil {
		run = execCLI
	}
	return run(ctx, name, args...)
}

func probeAgyCLI(ctx context.Context, run cliRunner, cmd string) cliCapabilities {
	if run == nil {
		run = execCLI
	}
	var caps cliCapabilities
	versionCtx, cancelVersion := context.WithTimeout(ctx, agyProbeTimeout)
	out, errOut, _ := run(versionCtx, cmd, "--version")
	cancelVersion()
	if strings.TrimSpace(out+errOut) != "" {
		caps.Version = parseAgyVersion(out + "\n" + errOut)
	}
	helpCtx, cancelHelp := context.WithTimeout(ctx, agyProbeTimeout)
	out, errOut, err := run(helpCtx, cmd, "--help")
	cancelHelp()
	if err == nil || strings.TrimSpace(out+errOut) != "" {
		text := out + "\n" + errOut
		caps.Model = helpHasFlag(text, "--model")
		caps.Effort = helpHasFlag(text, "--effort")
		caps.OutputFormat = helpHasFlag(text, "--output-format")
		caps.ModelsCommand = helpHasWord(text, "models")
	}
	if !caps.ModelsCommand && (caps.Model || versionAtLeast(caps.Version, 1, 0, 5)) {
		// agy models has existed since 1.0.5. Top-level --help often omits the
		// subcommand, and some builds print help with a non-zero exit.
		caps.ModelsCommand = true
	}
	caps.Probed = true
	caps.UsagePrint = caps.OutputFormat && versionAtLeast(caps.Version, 1, 1, 11)
	return caps
}

func (c cliCapabilities) canListModels() bool {
	if c.ModelsCommand || c.Model || versionAtLeast(c.Version, 1, 0, 5) {
		return true
	}
	// A probe that learned nothing still gets one documented `agy models` try.
	return c.Probed && c.Version == "" && !c.Model
}

func helpHasFlag(help, flag string) bool {
	for _, field := range strings.Fields(help) {
		field = strings.Trim(field, ",|[]()")
		if field == flag || strings.HasPrefix(field, flag+"=") {
			return true
		}
	}
	return false
}

func helpHasWord(help, word string) bool {
	for _, field := range strings.Fields(help) {
		field = strings.Trim(field, ",|[]()`")
		if strings.EqualFold(field, word) {
			return true
		}
	}
	return false
}

func parseAgyVersion(text string) string {
	for _, field := range strings.Fields(text) {
		field = strings.TrimPrefix(field, "v")
		field = strings.Trim(field, ",;()")
		if versionAtLeast(field, 0, 0, 0) && strings.Count(field, ".") >= 2 {
			parts := strings.SplitN(field, ".", 3)
			if len(parts) == 3 && isDigits(parts[0]) && isDigits(parts[1]) && isDigits(trimPreRelease(parts[2])) {
				return field
			}
		}
	}
	return ""
}

func trimPreRelease(s string) string {
	for i, r := range s {
		if r < '0' || r > '9' {
			return s[:i]
		}
	}
	return s
}

func isDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

func versionAtLeast(version string, major, minor, patch int) bool {
	version = strings.TrimPrefix(strings.TrimSpace(version), "v")
	if version == "" {
		return false
	}
	var got [3]int
	rest := version
	for i := 0; i < 3; i++ {
		part, next, _ := strings.Cut(rest, ".")
		part = trimPreRelease(part)
		if !isDigits(part) || part == "" {
			return false
		}
		n := 0
		for _, r := range part {
			n = n*10 + int(r-'0')
		}
		got[i] = n
		rest = next
	}
	if got[0] != major {
		return got[0] > major
	}
	if got[1] != minor {
		return got[1] > minor
	}
	return got[2] >= patch
}

func normalizeAgyEffort(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "low", "medium", "high", "xhigh", "max":
		return strings.ToLower(strings.TrimSpace(raw))
	default:
		return ""
	}
}

func splitEffortSuffix(id string) (base, effort string, ok bool) {
	id = strings.TrimSpace(id)
	lower := strings.ToLower(id)
	for _, suffix := range []string{"xhigh", "medium", "high", "low", "max"} {
		token := "-" + suffix
		if strings.HasSuffix(lower, token) && len(id) > len(token) {
			return id[:len(id)-len(token)], suffix, true
		}
	}
	return id, "", false
}

func effortsForModel(model string, catalog []core.ModelOption) []string {
	model = strings.TrimSpace(model)
	if model == "" || len(catalog) == 0 {
		return nil
	}
	base, _, _ := splitEffortSuffix(model)
	present := map[string]bool{}
	for _, item := range catalog {
		itemBase, effort, ok := splitEffortSuffix(item.Name)
		if !ok || !strings.EqualFold(itemBase, base) {
			continue
		}
		present[effort] = true
	}
	if len(present) == 0 {
		return nil
	}
	var out []string
	for _, effort := range agyEffortOrder {
		if present[effort] {
			out = append(out, effort)
		}
	}
	return out
}

func catalogHasExact(catalog []core.ModelOption, model string) bool {
	model = strings.TrimSpace(model)
	if model == "" {
		return false
	}
	for _, item := range catalog {
		if strings.EqualFold(strings.TrimSpace(item.Name), model) {
			return true
		}
	}
	return false
}

func variantID(base, effort string, catalog []core.ModelOption) string {
	want := strings.TrimSpace(base) + "-" + effort
	for _, item := range catalog {
		if strings.EqualFold(item.Name, want) {
			return item.Name
		}
	}
	return ""
}

type agyLaunch struct {
	Model  string
	Effort string
}

func resolveAgyLaunch(model, effort string, caps cliCapabilities, catalog []core.ModelOption) (agyLaunch, error) {
	model = strings.TrimSpace(model)
	effort = normalizeAgyEffort(effort)
	base, encoded, hasEncoded := splitEffortSuffix(model)
	known := effortsForModel(model, catalog)

	if hasEncoded && effort != "" && effort != encoded {
		sibling := variantID(base, effort, catalog)
		if sibling == "" {
			return agyLaunch{}, fmt.Errorf("%w: %s has no %q variant", errInvalidAgyCombo, model, effort)
		}
		model = sibling
	}

	if !hasEncoded && effort != "" {
		if sibling := variantID(model, effort, catalog); sibling != "" {
			model = sibling
			effort = ""
			hasEncoded = true
		} else if len(known) > 0 || catalogHasExact(catalog, model) {
			return agyLaunch{}, fmt.Errorf("%w: %s has no %q effort", errInvalidAgyCombo, model, effort)
		}
	}

	var launch agyLaunch
	if model != "" {
		if !caps.Model {
			return agyLaunch{}, nil
		}
		launch.Model = model
	}
	if effort != "" && !hasEncoded {
		if !caps.Effort {
			return agyLaunch{}, fmt.Errorf("%w: --effort is not advertised", errInvalidAgyCombo)
		}
		launch.Effort = effort
	}
	return launch, nil
}

func parseAgyModelCatalog(stdout string) ([]core.ModelOption, error) {
	stdout = strings.TrimSpace(stdout)
	if stdout == "" {
		return nil, fmt.Errorf("agy models returned no output")
	}
	if models, ok := parseAgyModelJSON(stdout); ok && len(models) > 0 {
		return models, nil
	}
	if models := parseAgyModelText(stdout); len(models) > 0 {
		return models, nil
	}
	return nil, fmt.Errorf("agy models output was not a model catalog")
}

func parseAgyModelJSON(stdout string) ([]core.ModelOption, bool) {
	payload := extractJSONObject(stdout)
	if payload == "" {
		return nil, false
	}
	var envelope struct {
		Response string `json:"response"`
		Command  struct {
			Name string `json:"name"`
			Data struct {
				Models []agyModelJSON `json:"models"`
			} `json:"data"`
		} `json:"command"`
	}
	if err := json.Unmarshal([]byte(payload), &envelope); err != nil {
		return nil, false
	}
	if len(envelope.Command.Data.Models) > 0 {
		return dedupeModels(envelope.Command.Data.Models), true
	}
	if strings.EqualFold(envelope.Command.Name, "models") && strings.TrimSpace(envelope.Response) != "" {
		if models := parseAgyModelText(envelope.Response); len(models) > 0 {
			return models, true
		}
	}
	return nil, false
}

type agyModelJSON struct {
	ID    string `json:"id"`
	Label string `json:"label"`
}

func dedupeModels(items []agyModelJSON) []core.ModelOption {
	seen := map[string]bool{}
	var models []core.ModelOption
	for _, item := range items {
		id := strings.TrimSpace(item.ID)
		if id == "" || seen[strings.ToLower(id)] {
			continue
		}
		seen[strings.ToLower(id)] = true
		models = append(models, core.ModelOption{Name: id, Desc: strings.TrimSpace(item.Label)})
	}
	return models
}

var modelIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]*$`)

func parseAgyModelText(text string) []core.ModelOption {
	seen := map[string]bool{}
	var models []core.ModelOption
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		id, label := splitModelLine(line)
		if !modelIDPattern.MatchString(id) || seen[strings.ToLower(id)] {
			continue
		}
		seen[strings.ToLower(id)] = true
		models = append(models, core.ModelOption{Name: id, Desc: label})
	}
	return models
}

func splitModelLine(line string) (id, label string) {
	if strings.Contains(line, "\t") {
		parts := strings.SplitN(line, "\t", 2)
		id = strings.TrimSpace(parts[0])
		if len(parts) == 2 {
			label = strings.TrimSpace(parts[1])
		}
		return id, label
	}
	fields := strings.Fields(line)
	if len(fields) == 0 {
		return "", ""
	}
	id = fields[0]
	if len(fields) > 1 {
		label = strings.TrimSpace(strings.TrimPrefix(line, id))
	}
	return id, label
}

func extractJSONObject(text string) string {
	start := strings.Index(text, "{")
	end := strings.LastIndex(text, "}")
	if start < 0 || end <= start {
		return ""
	}
	return text[start : end+1]
}

func (a *Agent) loadModelCatalog(ctx context.Context) ([]core.ModelOption, error) {
	caps, cached, fresh := a.catalogSnapshot()
	if fresh {
		return cached, nil
	}
	if !caps.canListModels() {
		if len(cached) > 0 {
			return cached, nil
		}
		return nil, fmt.Errorf("installed agy does not advertise a models command")
	}
	stdout, stderr, err := a.runModelsCommand(ctx, caps)
	if err != nil {
		if len(cached) > 0 {
			return cached, nil
		}
		return nil, fmt.Errorf("agy models: %s", sanitizeCLIText(joinCLIError(stderr, err)))
	}
	models, parseErr := parseAgyModelCatalog(stdout)
	if parseErr != nil {
		if len(cached) > 0 {
			return cached, nil
		}
		return nil, parseErr
	}
	a.rememberCatalog(models)
	return models, nil
}

func (a *Agent) catalogSnapshot() (cliCapabilities, []core.ModelOption, bool) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	if a.launch == nil {
		return cliCapabilities{}, nil, false
	}
	a.launch.mu.RLock()
	defer a.launch.mu.RUnlock()
	fresh := len(a.launch.catalog) > 0 && time.Since(a.launch.catalogAt) < agyCatalogTTL
	return a.launch.caps, append([]core.ModelOption(nil), a.launch.catalog...), fresh
}

func (a *Agent) runModelsCommand(ctx context.Context, caps cliCapabilities) (string, string, error) {
	if caps.OutputFormat {
		stdout, stderr, err := a.runCLI(ctx, "--output-format", "json", "models")
		if err == nil && strings.TrimSpace(stdout) != "" {
			if _, parseErr := parseAgyModelCatalog(stdout); parseErr == nil {
				return stdout, stderr, nil
			}
		}
	}
	return a.runCLI(ctx, "models")
}

func (a *Agent) rememberCatalog(models []core.ModelOption) {
	if len(models) == 0 {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	launch := a.ensureLaunchLocked()
	launch.mu.Lock()
	launch.catalog = append([]core.ModelOption(nil), models...)
	launch.catalogAt = time.Now()
	launch.mu.Unlock()
}

func joinCLIError(stderr string, err error) string {
	stderr = strings.TrimSpace(stderr)
	if err == nil {
		return stderr
	}
	if stderr == "" {
		return err.Error()
	}
	return err.Error() + ": " + stderr
}
