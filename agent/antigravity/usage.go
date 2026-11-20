package antigravity

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"strings"
	"time"

	"github.com/chenhg5/cc-connect/core"
)

const agyUsageTimeout = 20 * time.Second

func (a *Agent) UsageQueryTimeout() time.Duration { return agyUsageTimeout }

func (a *Agent) ModelCatalogTimeout() time.Duration { return agyUsageTimeout }

func (a *Agent) GetUsage(ctx context.Context) (*core.UsageReport, error) {
	if report, ok := a.cachedUsage(); ok {
		return report, nil
	}
	if deadline, ok := ctx.Deadline(); ok && time.Until(deadline) < agyUsageMinBudget {
		return nil, fmt.Errorf("agy quota query skipped: deadline too short")
	}

	_, _, caps, _ := a.launchSnapshot()
	if !caps.UsagePrint {
		return nil, fmt.Errorf("%w", core.ErrUsageUnsupported)
	}

	stdout, stderr, err := a.runCLI(ctx, "--output-format", "json", "-p", "/usage")
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return nil, fmt.Errorf("agy quota query timed out")
		}
		text := sanitizeCLIText(joinCLIError(stderr, err))
		if usageCommandMissing(text) {
			return nil, fmt.Errorf("%w", core.ErrUsageUnsupported)
		}
		return nil, fmt.Errorf("agy quota query failed: %s", text)
	}

	report, parseErr := parseAgyUsageReport(stdout, time.Now())
	if parseErr != nil {
		if errors.Is(parseErr, core.ErrUsageUnsupported) {
			return nil, parseErr
		}
		return nil, fmt.Errorf("agy quota query failed: %s", sanitizeCLIText(parseErr.Error()))
	}
	a.storeUsage(report)
	return report, nil
}

func (a *Agent) cachedUsage() (*core.UsageReport, bool) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	if a.usageReport == nil || time.Since(a.usageAt) >= agyCatalogTTL {
		return nil, false
	}
	return a.usageReport, true
}

func (a *Agent) storeUsage(report *core.UsageReport) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.usageReport = report
	a.usageAt = time.Now()
}

func usageCommandMissing(text string) bool {
	lower := strings.ToLower(text)
	return strings.Contains(lower, "unknown command") ||
		strings.Contains(lower, "not recognized") ||
		strings.Contains(lower, "takes no arguments") && strings.Contains(lower, "prompt")
}

type agyUsageEnvelope struct {
	Status   string `json:"status"`
	Error    string `json:"error"`
	NumTurns int    `json:"num_turns"`
	Usage    struct {
		OutputTokens int `json:"output_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
	Command struct {
		Name string          `json:"name"`
		Data json.RawMessage `json:"data"`
	} `json:"command"`
}

type agyUsageBucketJSON struct {
	ID                string   `json:"id"`
	Name              string   `json:"name"`
	Window            string   `json:"window"`
	RemainingFraction *float64 `json:"remaining_fraction"`
	ResetTime         string   `json:"reset_time"`
}

type agyUsageGroupJSON struct {
	Name    string               `json:"name"`
	Buckets []agyUsageBucketJSON `json:"buckets"`
}

type agyUsageData struct {
	Groups []agyUsageGroupJSON `json:"groups"`
}

func parseAgyUsageReport(stdout string, now time.Time) (*core.UsageReport, error) {
	payload := extractJSONObject(stdout)
	if payload == "" {
		return nil, fmt.Errorf("%w", core.ErrUsageUnsupported)
	}
	var envelope agyUsageEnvelope
	if err := json.Unmarshal([]byte(payload), &envelope); err != nil {
		return nil, fmt.Errorf("%w", core.ErrUsageUnsupported)
	}
	if envelope.NumTurns > 0 || envelope.Usage.OutputTokens > 0 || envelope.Usage.TotalTokens > 0 {
		return nil, fmt.Errorf("%w", core.ErrUsageUnsupported)
	}
	name := strings.ToLower(strings.TrimSpace(envelope.Command.Name))
	if name != "usage" && name != "quota" {
		if strings.EqualFold(envelope.Status, "ERROR") && usageCommandMissing(envelope.Error) {
			return nil, fmt.Errorf("%w", core.ErrUsageUnsupported)
		}
		return nil, fmt.Errorf("%w", core.ErrUsageUnsupported)
	}
	if strings.EqualFold(envelope.Status, "ERROR") {
		return nil, fmt.Errorf("agy quota query failed: %s", sanitizeCLIText(envelope.Error))
	}

	var data agyUsageData
	if len(envelope.Command.Data) == 0 || json.Unmarshal(envelope.Command.Data, &data) != nil || len(data.Groups) == 0 {
		return nil, fmt.Errorf("%w", core.ErrUsageUnsupported)
	}

	report := &core.UsageReport{Provider: "antigravity"}
	for _, group := range data.Groups {
		bucket, ok := agyGroupBucket(group.Name, group.Buckets, now)
		if !ok {
			continue
		}
		report.Buckets = append(report.Buckets, bucket)
	}
	if len(report.Buckets) == 0 {
		return nil, fmt.Errorf("%w", core.ErrUsageUnsupported)
	}
	return report, nil
}

func agyGroupBucket(group string, buckets []agyUsageBucketJSON, now time.Time) (core.UsageBucket, bool) {
	var windows []core.UsageWindow
	limitReached := false
	allowed := false
	for _, item := range buckets {
		if item.RemainingFraction == nil {
			continue
		}
		seconds, ok := agyWindowSeconds(item.Window, item.ID, item.Name)
		if !ok {
			continue
		}
		remaining := *item.RemainingFraction
		if remaining < 0 {
			remaining = 0
		}
		if remaining > 1 {
			remaining = 1
		}
		remainingPct := int(math.Round(remaining * 100))
		window := core.UsageWindow{
			Name:          firstNonEmpty(item.Name, item.ID),
			UsedPercent:   100 - remainingPct,
			WindowSeconds: seconds,
		}
		if reset, err := time.Parse(time.RFC3339, strings.TrimSpace(item.ResetTime)); err == nil {
			window.ResetAtUnix = reset.Unix()
			if reset.After(now) {
				window.ResetAfterSeconds = int(reset.Sub(now).Seconds())
			}
		}
		if remaining > 0 {
			allowed = true
		} else {
			limitReached = true
		}
		windows = append(windows, window)
	}
	if len(windows) == 0 {
		return core.UsageBucket{}, false
	}
	return core.UsageBucket{
		Name:         group,
		Allowed:      allowed,
		LimitReached: limitReached && !allowed,
		Windows:      windows,
	}, true
}

func agyWindowSeconds(window, id, name string) (int, bool) {
	text := strings.ToLower(window + " " + id + " " + name)
	switch {
	case strings.Contains(text, "5h") || strings.Contains(text, "five hour") || strings.Contains(text, "5-hour") || strings.Contains(text, "5 hour"):
		return 18000, true
	case strings.Contains(text, "weekly") || strings.Contains(text, "week"):
		return 604800, true
	default:
		return 0, false
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

var (
	reBearer = regexp.MustCompile(`(?i)bearer\s+\S+`)
	reSecret = regexp.MustCompile(`(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*\S+`)
	reYa29   = regexp.MustCompile(`ya29\.[A-Za-z0-9._\-]+`)
	reSK     = regexp.MustCompile(`sk-[A-Za-z0-9]+`)
)

func sanitizeCLIText(text string) string {
	text = strings.Join(strings.Fields(text), " ")
	if text == "" {
		return "unknown error"
	}
	text = reBearer.ReplaceAllString(text, "bearer [redacted]")
	text = reSecret.ReplaceAllString(text, "$1=[redacted]")
	text = reYa29.ReplaceAllString(text, "[redacted]")
	text = reSK.ReplaceAllString(text, "[redacted]")
	if len(text) > 240 {
		text = text[:240]
	}
	return text
}

func (a *Agent) launchSnapshot() (model, effort string, caps cliCapabilities, catalog []core.ModelOption) {
	a.mu.RLock()
	launch := a.launch
	a.mu.RUnlock()
	if launch == nil {
		a.mu.RLock()
		model = a.model
		a.mu.RUnlock()
		return model, "", cliCapabilities{}, nil
	}
	return launch.snapshot()
}
