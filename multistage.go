// This is free and unencumbered software released into the public domain.
//
// Anyone is free to copy, modify, publish, use, compile, sell, or
// distribute this software, either in source code form or as a compiled
// binary, for any purpose, commercial or non-commercial, and by any
// means.
//
// In jurisdictions that recognize copyright laws, the author or authors
// of this software dedicate any and all copyright interest in the
// software to the public domain. We make this dedication for the benefit
// of the public at large and to the detriment of our heirs and
// successors. We intend this dedication to be an overt act of
// relinquishment in perpetuity of all present and future rights to this
// software under copyright law.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// For more information, please refer to <http://unlicense.org/>

package gomplerate

import (
	"fmt"
	"math"
)

// This file implements multi-stage resampling. Instead of running a single
// interpolation/filtering pass at the least common multiple of the input and
// output rates, the conversion ratio is factored into a reproducible sequence
// of smaller interpolate/decimate stages. Every stage zero-stuffs (on
// interpolation), applies a zero-phase windowed-sinc low-pass filter and
// picks every m-th sample (on decimation). Splitting the ratio keeps both
// the per-stage filter length and the image/alias frequencies manageable,
// which is what you want when previewing low sample rate audio at a much
// higher rate (or the other way around).

const (
	// maxStageFactor is the largest interpolate/decimate factor a single
	// stage will use. Bigger prime factors of the ratio get their own stage.
	maxStageFactor = 16
	// maxStages bounds the number of stages in a plan.
	maxStages = 64
	// minHalfLen is the minimum number of filter taps on each side of the
	// center tap.
	minHalfLen = 24
	// maxHalfLen caps the filter half-length so that stages with huge
	// factors (e.g. a large prime ratio) don't allocate gigantic filters.
	// The transition band widens accordingly for such ratios.
	maxHalfLen = 4095
	// factorTrialLimit bounds trial division when factoring a ratio. Any
	// unfactored remainder is used as a single stage factor; this keeps
	// pathological rates (huge primes) fast at the cost of a wider
	// transition band for that stage.
	factorTrialLimit = 1 << 20
)

// Stage describes a single low-pass + resample stage of a multi-stage plan.
type Stage struct {
	Interpolate int // Upsampling factor of the stage (1 = none).
	Decimate    int // Downsampling factor of the stage (1 = none).
	FromRate    int // Sample rate at the input of the stage.
	ToRate      int // Sample rate at the output of the stage.
}

// stageFilter is a Stage with its computed FIR filter.
type stageFilter struct {
	l, m int       // interpolate / decimate factors
	rate int       // sample rate after this stage
	half int       // half length of taps
	taps []float64 // windowed-sinc low-pass, zero-phase, DC gain == l
}

// PlanStages returns the deterministic, reproducible sequence of stages used
// to convert from fromRate to toRate. The product of the stage ratios equals
// toRate/fromRate exactly, and the output rate of every stage feeds the input
// rate of the next. It returns nil (and no error) when the rates are equal.
//
// The plan keeps intermediate rates at or below max(fromRate, toRate)
// whenever the ratio allows it, and never lets them drop below
// min(fromRate, toRate), which would destroy part of the passband. When the
// factorization forces a choice, the plan overshoots above the endpoint
// rates (never dips below them). An error is returned for non-positive
// rates and for ratios whose intermediate rate would overflow an int.
func PlanStages(fromRate, toRate int) ([]Stage, error) {
	if fromRate < 1 {
		return nil, fmt.Errorf("input sample rate must be bigger than 0 (got %d)", fromRate)
	}
	if toRate < 1 {
		return nil, fmt.Errorf("output sample rate must be bigger than 0 (got %d)", toRate)
	}
	if fromRate == toRate {
		return nil, nil
	}

	g := gcd(fromRate, toRate)
	l := toRate / g   // total interpolation factor
	m := fromRate / g // total decimation factor

	// The highest possible intermediate rate is fromRate*l == toRate*m;
	// make sure it cannot overflow before planning anything.
	if int64(fromRate) > math.MaxInt64/int64(l) {
		return nil, fmt.Errorf("resampling ratio %d/%d is too extreme (intermediate rate overflow)", toRate, fromRate)
	}

	lf := mergeFactors(factorize(l), maxStageFactor)
	mf := mergeFactors(factorize(m), maxStageFactor)
	if len(lf)+len(mf) > maxStages {
		return nil, fmt.Errorf("resampling ratio %d/%d requires too many stages (%d)", toRate, fromRate, len(lf)+len(mf))
	}

	lo, hi := fromRate, toRate
	if lo > hi {
		lo, hi = hi, lo
	}

	stages := make([]Stage, 0, len(lf)+len(mf))
	cur := fromRate
	i, j := 0, 0
	interp := func() {
		stages = append(stages, Stage{Interpolate: lf[i], Decimate: 1, FromRate: cur, ToRate: cur * lf[i]})
		cur *= lf[i]
		i++
	}
	decimate := func() {
		stages = append(stages, Stage{Interpolate: 1, Decimate: mf[j], FromRate: cur, ToRate: cur / mf[j]})
		cur /= mf[j]
		j++
	}
	for i < len(lf) || j < len(mf) {
		switch {
		case i < len(lf) && cur <= hi/lf[i]:
			// Interpolation keeps the intermediate rate at or below the
			// larger endpoint rate (division form avoids overflow).
			interp()
		case j < len(mf) && cur/mf[j] >= lo:
			// Decimation keeps the intermediate rate at or above the
			// smaller endpoint rate, preserving the passband.
			decimate()
		case i < len(lf):
			// Forced overshoot above hi: the remaining decimations would
			// dip below lo, so the interpolation has to happen first.
			interp()
		default:
			// Only decimations left; these telescope down to toRate and
			// can never dip below lo.
			decimate()
		}
	}
	return stages, nil
}

// NewMultiStageResampler creates a Resampler that converts between the given
// rates using a multi-stage FIR pipeline (see PlanStages) instead of the
// default single-pass spline resampler. The returned Resampler is used
// exactly like one created with NewResampler.
func NewMultiStageResampler(channels, inputRate, outputRate int) (*Resampler, error) {
	resampler, err := NewResampler(channels, inputRate, outputRate)
	if err != nil {
		return nil, err
	}
	if err := resampler.UseMultiStage(); err != nil {
		return nil, err
	}
	return resampler, nil
}

// UseMultiStage switches this resampler to the multi-stage FIR pipeline
// (see PlanStages). ResampleFloat64 and ResampleInt16 keep working as before;
// only the conversion algorithm changes. Call it again (or never call it) to
// stay on the legacy single-pass spline path.
func (resampler *Resampler) UseMultiStage() error {
	plan, err := PlanStages(resampler.FromRate, resampler.ToRate)
	if err != nil {
		return err
	}
	stages := make([]stageFilter, 0, len(plan))
	for _, s := range plan {
		taps, half := designStageFilter(s.Interpolate, s.Decimate)
		stages = append(stages, stageFilter{
			l:    s.Interpolate,
			m:    s.Decimate,
			rate: s.ToRate,
			half: half,
			taps: taps,
		})
	}
	// A non-nil (possibly empty) slice selects the multi-stage path.
	resampler.stages = stages
	return nil
}

// Stages returns the multi-stage plan this resampler was configured with, or
// nil if the multi-stage path is not enabled.
func (resampler *Resampler) Stages() []Stage {
	if resampler.stages == nil {
		return nil
	}
	stages := make([]Stage, 0, len(resampler.stages))
	cur := resampler.FromRate
	for _, st := range resampler.stages {
		next := cur
		if st.m == 1 {
			next = cur * st.l
		} else {
			next = cur / st.m
		}
		stages = append(stages, Stage{Interpolate: st.l, Decimate: st.m, FromRate: cur, ToRate: next})
		cur = next
	}
	return stages
}

// resampleMultiStage resamples interleaved audio through the stage pipeline.
//
// Length and edge-case contract:
//   - empty input returns nil;
//   - input shorter than one frame (len(data) < Channels) returns nil;
//   - a trailing incomplete frame is ignored;
//   - the output holds exactly Channels*round(frames*ToRate/FromRate) samples,
//     which may be 0 for very short inputs with a large downsample ratio;
//   - inputs shorter than the filters are implicitly zero-padded;
//   - if the output length would overflow, nil is returned.
func (resampler *Resampler) resampleMultiStage(data []float64) []float64 {
	if len(data) == 0 {
		return nil
	}
	if len(resampler.stages) == 0 {
		return data[:]
	}
	nFrames := len(data) / resampler.Channels
	if nFrames == 0 {
		return nil
	}

	// Per-stage output lengths are derived from the global time axis
	// (frames at FromRate mapped onto each stage's rate), so every stage
	// stays phase-aligned with the input regardless of rounding.
	lens := make([]int, len(resampler.stages))
	for si := range resampler.stages {
		n, ok := stageLen(nFrames, resampler.FromRate, resampler.stages[si].rate)
		if !ok {
			return nil // length overflow
		}
		if n < 1 && si < len(resampler.stages)-1 {
			n = 1 // keep intermediate stages from collapsing the signal
		}
		lens[si] = n
	}
	target := lens[len(lens)-1]

	out := make([]float64, target*resampler.Channels)
	ch := make([]float64, nFrames)
	var bufs [2][]float64 // ping-pong scratch, reused across channels and stages

	for c := 0; c < resampler.Channels; c++ {
		for i := 0; i < nFrames; i++ {
			ch[i] = data[i*resampler.Channels+c]
		}
		cur := ch
		for si := range resampler.stages {
			st := &resampler.stages[si]
			outLen := lens[si]
			buf := bufs[si%2]
			if cap(buf) < outLen {
				buf = make([]float64, outLen)
			}
			buf = buf[:outLen]
			st.process(cur, buf)
			bufs[si%2] = buf
			cur = buf
		}
		// Interleave this channel into the output. cur always has target
		// samples by construction; clamp defensively.
		n := target
		if len(cur) < n {
			n = len(cur)
		}
		for i := 0; i < n; i++ {
			out[i*resampler.Channels+c] = cur[i]
		}
	}
	return out
}

// process computes out[k] = (in upsampled by l, low-pass filtered)[k*m] for
// every k, without materializing the upsampled buffer. The filter is
// zero-phase: output sample k aligns with input time k*m/l.
func (st *stageFilter) process(in, out []float64) {
	l, m, half := st.l, st.m, st.half
	h := st.taps
	n := len(in)
	nh := len(h)
	for k := range out {
		center := half + k*m
		acc := 0.0
		// The upsampled signal is non-zero only at multiples of l, so the
		// convolution only needs taps with (center-j) % l == 0.
		for j := center % l; j < nh; j += l {
			t := (center - j) / l
			if t < 0 {
				break
			}
			if t < n {
				acc += in[t] * h[j]
			}
		}
		out[k] = acc
	}
}

// designStageFilter builds a zero-phase windowed-sinc low-pass filter for a
// stage with the given interpolate/decimate factors. The cutoff sits below
// the lower of the two Nyquist limits of the stage, and the DC gain is l so
// that zero-stuffed interpolation preserves amplitude.
func designStageFilter(l, m int) (taps []float64, halfLen int) {
	f := l
	if m > f {
		f = m
	}
	halfLen = 24 * f
	if halfLen < minHalfLen {
		halfLen = minHalfLen
	}
	if halfLen > maxHalfLen {
		halfLen = maxHalfLen
	}
	n := 2*halfLen + 1
	fc := 0.42 / float64(f) // cycles/sample at the stage's high rate

	taps = make([]float64, n)
	for i := 0; i < n; i++ {
		x := float64(i - halfLen)
		var s float64
		if x == 0 {
			s = 2 * fc
		} else {
			s = math.Sin(2*math.Pi*fc*x) / (math.Pi * x)
		}
		// Blackman window.
		w := 0.42 -
			0.5*math.Cos(2*math.Pi*float64(i)/float64(n-1)) +
			0.08*math.Cos(4*math.Pi*float64(i)/float64(n-1))
		taps[i] = s * w
	}
	// Normalize every polyphase branch to unit DC gain, so a constant
	// signal comes out constant at every interpolation phase (for
	// decimation stages this is a plain sum normalization).
	for p := 0; p < l; p++ {
		sum := 0.0
		for j := p; j < n; j += l {
			sum += taps[j]
		}
		if sum == 0 {
			continue
		}
		for j := p; j < n; j += l {
			taps[j] /= sum
		}
	}
	return taps, halfLen
}

// stageLen maps n input frames at fromRate onto rate, rounding to the
// nearest sample. It reports false if the length cannot be represented.
func stageLen(n, fromRate, rate int) (int, bool) {
	est := float64(n) * float64(rate) / float64(fromRate)
	if math.IsNaN(est) || math.IsInf(est, 0) || est > float64(math.MaxInt64)/2 {
		return 0, false
	}
	return int(math.Round(est)), true
}

// gcd returns the greatest common divisor of a and b (both must be > 0).
func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// factorize returns the prime factors of n in ascending order. Trial
// division is bounded by factorTrialLimit; any larger unfactored remainder
// is returned as a single final factor (it may be composite for
// astronomically large rates, which only affects filter sharpness, not
// correctness of the ratio).
func factorize(n int) []int {
	var out []int
	for n%2 == 0 {
		out = append(out, 2)
		n /= 2
	}
	for f := 3; f <= n/f && f <= factorTrialLimit; f += 2 {
		for n%f == 0 {
			out = append(out, f)
			n /= f
		}
	}
	if n > 1 {
		out = append(out, n)
	}
	return out
}

// mergeFactors greedily multiplies adjacent ascending factors into stage
// factors that do not exceed cap. Factors already larger than cap are kept
// as their own stage.
func mergeFactors(factors []int, cap int) []int {
	var out []int
	acc := 1
	for _, f := range factors {
		if f > cap {
			if acc > 1 {
				out = append(out, acc)
				acc = 1
			}
			out = append(out, f)
			continue
		}
		if acc*f > cap {
			out = append(out, acc)
			acc = f
		} else {
			acc *= f
		}
	}
	if acc > 1 {
		out = append(out, acc)
	}
	return out
}
