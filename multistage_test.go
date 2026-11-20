// This is free and unencumbered software released into the public domain.
// For more information, please refer to <http://unlicense.org/>

package gomplerate

import (
	"math"
	"reflect"
	"testing"
)

// --- helpers ---------------------------------------------------------------

func sine(freq float64, rate, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = math.Sin(2 * math.Pi * freq * float64(i) / float64(rate))
	}
	return out
}

func chirp(f0, f1 float64, rate, n int) []float64 {
	out := make([]float64, n)
	k := (f1 - f0) / (float64(n) / float64(rate)) // Hz per second
	phase := 0.0
	for i := range out {
		t := float64(i) / float64(rate)
		phase = 2 * math.Pi * (f0*t + 0.5*k*t*t)
		out[i] = math.Sin(phase)
	}
	return out
}

func rms(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	acc := 0.0
	for _, v := range data {
		acc += v * v
	}
	return math.Sqrt(acc / float64(len(data)))
}

// dftMag returns the magnitude of the component at freq Hz, normalized so a
// full-scale sine measures ~1.
func dftMag(data []float64, rate int, freq float64) float64 {
	re, im := 0.0, 0.0
	for i, v := range data {
		a := 2 * math.Pi * freq * float64(i) / float64(rate)
		re += v * math.Cos(a)
		im -= v * math.Sin(a)
	}
	return 2 * math.Sqrt(re*re+im*im) / float64(len(data))
}

func argmax(data []float64) int {
	best, bi := -math.MaxFloat64, 0
	for i, v := range data {
		if v > best {
			best, bi = v, i
		}
	}
	return bi
}

func interleave(channels ...[]float64) []float64 {
	n := len(channels[0])
	out := make([]float64, n*len(channels))
	for c, ch := range channels {
		for i := 0; i < n; i++ {
			out[i*len(channels)+c] = ch[i]
		}
	}
	return out
}

func deinterleave(data []float64, channels, c int) []float64 {
	n := len(data) / channels
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = data[i*channels+c]
	}
	return out
}

// --- plan tests ------------------------------------------------------------

func TestPlanStagesDeterministicAndConsistent(t *testing.T) {
	pairs := [][2]int{
		{8000, 48000}, {48000, 8000}, {44100, 48000}, {48000, 44100},
		{11000, 44100}, {11025, 44100}, {32000, 16000}, {8000, 8000},
		{2, 8191}, {8000, 1},
	}
	for _, p := range pairs {
		from, to := p[0], p[1]
		plan1, err := PlanStages(from, to)
		if err != nil {
			t.Fatalf("PlanStages(%d, %d): %v", from, to, err)
		}
		plan2, err := PlanStages(from, to)
		if err != nil {
			t.Fatalf("PlanStages(%d, %d) second call: %v", from, to, err)
		}
		if !reflect.DeepEqual(plan1, plan2) {
			t.Fatalf("PlanStages(%d, %d) not reproducible:\n%v\n%v", from, to, plan1, plan2)
		}
		if from == to {
			if len(plan1) != 0 {
				t.Fatalf("PlanStages(%d, %d) = %v, want no stages", from, to, plan1)
			}
			continue
		}
		// Chain consistency and exact overall ratio.
		if plan1[0].FromRate != from {
			t.Errorf("PlanStages(%d, %d): first stage FromRate = %d", from, to, plan1[0].FromRate)
		}
		prodL, prodM := 1, 1
		for i, s := range plan1 {
			if i > 0 && s.FromRate != plan1[i-1].ToRate {
				t.Errorf("PlanStages(%d, %d): stage %d input rate %d != previous output %d",
					from, to, i, s.FromRate, plan1[i-1].ToRate)
			}
			if s.Interpolate < 1 || s.Decimate < 1 {
				t.Errorf("PlanStages(%d, %d): stage %d has invalid factors %+v", from, to, i, s)
			}
			if s.ToRate != s.FromRate*s.Interpolate/s.Decimate {
				t.Errorf("PlanStages(%d, %d): stage %d rate mismatch %+v", from, to, i, s)
			}
			prodL *= s.Interpolate
			prodM *= s.Decimate
		}
		if plan1[len(plan1)-1].ToRate != to {
			t.Errorf("PlanStages(%d, %d): last stage ToRate = %d", from, to, plan1[len(plan1)-1].ToRate)
		}
		if from*prodL != to*prodM {
			t.Errorf("PlanStages(%d, %d): ratio mismatch %d*%d != %d*%d", from, to, from, prodL, to, prodM)
		}
		// Intermediate rates must never drop below the smaller endpoint
		// rate, or part of the passband would be destroyed.
		lo := from
		if to < lo {
			lo = to
		}
		for i, s := range plan1 {
			if s.ToRate < lo {
				t.Errorf("PlanStages(%d, %d): stage %d dips to %d (< %d)", from, to, i, s.ToRate, lo)
			}
		}
	}
}

func TestPlanStagesInvalidRates(t *testing.T) {
	for _, p := range [][2]int{{0, 100}, {100, 0}, {-1, 100}, {100, -2}, {0, 0}} {
		if _, err := PlanStages(p[0], p[1]); err == nil {
			t.Errorf("PlanStages(%d, %d): expected error", p[0], p[1])
		}
	}
}

func TestPlanStagesOverflow(t *testing.T) {
	// Consecutive huge integers are coprime, so the intermediate rate
	// would be from*to, which overflows int64.
	if _, err := PlanStages(1<<62, (1<<62)+1); err == nil {
		t.Fatal("PlanStages with overflowing intermediate rate: expected error")
	}
}

func TestNewMultiStageResamplerErrors(t *testing.T) {
	cases := [][3]int{
		{0, 8000, 16000},  // no channels
		{-1, 8000, 16000}, // negative channels
		{1, 0, 16000},     // zero input rate
		{1, 8000, 0},      // zero output rate
		{1, -8000, 16000}, // negative input rate
		{1, 8000, -16000}, // negative output rate
	}
	for _, c := range cases {
		if _, err := NewMultiStageResampler(c[0], c[1], c[2]); err == nil {
			t.Errorf("NewMultiStageResampler(%d, %d, %d): expected error", c[0], c[1], c[2])
		}
	}
}

func TestStagesGetter(t *testing.T) {
	r, err := NewMultiStageResampler(2, 44100, 48000)
	if err != nil {
		t.Fatal(err)
	}
	got := r.Stages()
	want, err := PlanStages(44100, 48000)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Stages() = %v, want %v", got, want)
	}
	// Legacy resampler has no plan.
	legacy, err := NewResampler(2, 44100, 48000)
	if err != nil {
		t.Fatal(err)
	}
	if legacy.Stages() != nil {
		t.Fatal("Stages() on legacy resampler should be nil")
	}
}

// --- length and structure tests --------------------------------------------

func TestMultiStageLengths(t *testing.T) {
	cases := []struct {
		channels, from, to, frames int
	}{
		{1, 8000, 48000, 1000},
		{1, 48000, 8000, 4800},
		{2, 44100, 8000, 4410},
		{2, 8000, 44100, 800},
		{2, 11000, 44100, 24}, // README example size
		{4, 32000, 16000, 333},
		{1, 48000, 48000, 500},
		{3, 22050, 44100, 100},
		{1, 44100, 48000, 441},
	}
	for _, c := range cases {
		r, err := NewMultiStageResampler(c.channels, c.from, c.to)
		if err != nil {
			t.Fatal(err)
		}
		in := make([]float64, c.frames*c.channels)
		for i := range in {
			in[i] = math.Sin(float64(i) * 0.1)
		}
		out := r.ResampleFloat64(in)
		want := c.channels * int(math.Round(float64(c.frames)*float64(c.to)/float64(c.from)))
		if len(out) != want {
			t.Errorf("%dch %d->%d, %d frames: got %d samples, want %d",
				c.channels, c.from, c.to, c.frames, len(out), want)
		}
	}
}

func TestMultiStageEmptyAndPartialInput(t *testing.T) {
	r, err := NewMultiStageResampler(2, 8000, 48000)
	if err != nil {
		t.Fatal(err)
	}
	if got := r.ResampleFloat64(nil); got != nil {
		t.Errorf("nil input: got %v, want nil", got)
	}
	if got := r.ResampleFloat64([]float64{}); got != nil {
		t.Errorf("empty input: got %v, want nil", got)
	}
	// Less than one full frame.
	if got := r.ResampleFloat64([]float64{0.5}); got != nil {
		t.Errorf("partial frame: got %v, want nil", got)
	}
	// A trailing incomplete frame is ignored.
	full := r.ResampleFloat64(make([]float64, 20))
	withTail := r.ResampleFloat64(make([]float64, 21))
	if len(full) != len(withTail) {
		t.Errorf("trailing partial frame changed length: %d vs %d", len(full), len(withTail))
	}
}

func TestMultiStageShortInput(t *testing.T) {
	// Inputs shorter than the FIR filter must still work (zero padding).
	for _, n := range []int{1, 2, 3, 5, 16} {
		rUp, err := NewMultiStageResampler(1, 8000, 48000)
		if err != nil {
			t.Fatal(err)
		}
		in := make([]float64, n)
		in[0] = 1
		out := rUp.ResampleFloat64(in)
		want := int(math.Round(float64(n) * 48000.0 / 8000.0))
		if len(out) != want {
			t.Errorf("upsample %d samples: got %d, want %d", n, len(out), want)
		}

		rDown, err := NewMultiStageResampler(1, 48000, 8000)
		if err != nil {
			t.Fatal(err)
		}
		out = rDown.ResampleFloat64(in)
		want = int(math.Round(float64(n) * 8000.0 / 48000.0))
		if len(out) != want {
			t.Errorf("downsample %d samples: got %d, want %d", n, len(out), want)
		}
	}
}

func TestMultiStageExtremeRatios(t *testing.T) {
	// Heavy decimation: 8000 -> 1 Hz.
	r, err := NewMultiStageResampler(1, 8000, 1)
	if err != nil {
		t.Fatal(err)
	}
	out := r.ResampleFloat64(sine(0.5, 8000, 16000))
	if len(out) != 2 {
		t.Errorf("8000->1: got %d samples, want 2", len(out))
	}

	// Large prime ratio: single stage with capped filter length.
	r, err = NewMultiStageResampler(1, 2, 8191)
	if err != nil {
		t.Fatal(err)
	}
	in := make([]float64, 100)
	in[50] = 1
	out = r.ResampleFloat64(in)
	want := int(math.Round(100.0 * 8191.0 / 2.0))
	if len(out) != want {
		t.Errorf("2->8191: got %d samples, want %d", len(out), want)
	}
}

func TestStageLenOverflow(t *testing.T) {
	if _, ok := stageLen(math.MaxInt64/2, 1, 4); ok {
		t.Error("stageLen should report overflow")
	}
	if n, ok := stageLen(10, 2, 3); !ok || n != 15 {
		t.Errorf("stageLen(10, 2, 3) = %d, %v; want 15, true", n, ok)
	}
	if n, ok := stageLen(0, 2, 3); !ok || n != 0 {
		t.Errorf("stageLen(0, 2, 3) = %d, %v; want 0, true", n, ok)
	}
}

// --- signal tests ----------------------------------------------------------

func TestMultiStageAliasing(t *testing.T) {
	r, err := NewMultiStageResampler(1, 44100, 8000)
	if err != nil {
		t.Fatal(err)
	}
	n := 44100

	// 1 kHz is well inside the 4 kHz passband and must survive.
	pass := r.ResampleFloat64(sine(1000, 44100, n))
	got := rms(pass[1000 : len(pass)-1000])
	if math.Abs(got-0.7071) > 0.05 {
		t.Errorf("1 kHz passband RMS = %.4f, want ~0.707", got)
	}

	// 6 kHz is above the output Nyquist (4 kHz) and must be suppressed.
	stop := r.ResampleFloat64(sine(6000, 44100, n))
	got = rms(stop[1000 : len(stop)-1000])
	if got > 0.02 {
		t.Errorf("6 kHz alias RMS = %.4f, want < 0.02", got)
	}
}

func TestMultiStageSweep(t *testing.T) {
	// Linear sweep 100 Hz -> 20 kHz over 1 s at 44.1 kHz, resampled to
	// 8 kHz. Everything above the 4 kHz Nyquist limit (the last ~80% of
	// the sweep) must be filtered out, not folded back.
	r, err := NewMultiStageResampler(1, 44100, 8000)
	if err != nil {
		t.Fatal(err)
	}
	out := r.ResampleFloat64(chirp(100, 20000, 44100, 44100))
	if len(out) != 8000 {
		t.Fatalf("sweep output length = %d, want 8000", len(out))
	}
	early := rms(out[200:1200]) // ~100 Hz - 2.1 kHz, in passband
	late := rms(out[2400:7600]) // ~6.6 kHz - 19.4 kHz, must be gone
	if early < 0.5 {
		t.Errorf("passband sweep RMS = %.4f, want ~0.7", early)
	}
	if late > early/8 {
		t.Errorf("aliased sweep RMS = %.4f (passband %.4f), want > 8x suppression", late, early)
	}
}

func TestMultiStageUpsamplingImages(t *testing.T) {
	// 1 kHz sine at 8 kHz upsampled to 48 kHz: spectral images at
	// 8k-1k=7 kHz and 8k+1k=9 kHz must be suppressed by the low-pass.
	r, err := NewMultiStageResampler(1, 8000, 48000)
	if err != nil {
		t.Fatal(err)
	}
	out := r.ResampleFloat64(sine(1000, 8000, 8000))
	if len(out) != 48000 {
		t.Fatalf("output length = %d, want 48000", len(out))
	}
	steady := out[2400 : len(out)-2400]
	if m := dftMag(steady, 48000, 1000); math.Abs(m-1) > 0.05 {
		t.Errorf("1 kHz magnitude = %.4f, want ~1", m)
	}
	for _, f := range []float64{7000, 9000, 15000} {
		if m := dftMag(steady, 48000, f); m > 0.02 {
			t.Errorf("image at %.0f Hz: magnitude %.4f, want < 0.02", f, m)
		}
	}
}

func TestMultiStageImpulse(t *testing.T) {
	// Upsampling: impulse position, zero-phase symmetry.
	r, err := NewMultiStageResampler(1, 8000, 48000)
	if err != nil {
		t.Fatal(err)
	}
	n := 4000
	in := make([]float64, n)
	in[2000] = 1
	out := r.ResampleFloat64(in)
	if len(out) != 24000 {
		t.Fatalf("output length = %d, want 24000", len(out))
	}
	pk := argmax(out)
	if pk < 11998 || pk > 12002 {
		t.Errorf("impulse peak at %d, want ~12000 (zero group delay)", pk)
	}
	for d := 1; d <= 32; d++ {
		if math.Abs(out[pk+d]-out[pk-d]) > 0.05*out[pk] {
			t.Errorf("impulse response not symmetric at offset %d: %f vs %f", d, out[pk+d], out[pk-d])
		}
	}

	// Multi-stage downsampling: impulse position preserved.
	r, err = NewMultiStageResampler(1, 44100, 8000)
	if err != nil {
		t.Fatal(err)
	}
	n = 44100
	in = make([]float64, n)
	in[22050] = 1
	out = r.ResampleFloat64(in)
	if len(out) != 8000 {
		t.Fatalf("output length = %d, want 8000", len(out))
	}
	pk = argmax(out)
	want := 22050 * 8000 / 44100 // 4000
	if pk < want-2 || pk > want+2 {
		t.Errorf("impulse peak at %d, want ~%d", pk, want)
	}
}

func TestMultiStageChannels(t *testing.T) {
	// Stereo: left = 500 Hz sine, right = centered impulse.
	n := 4000
	left := sine(500, 8000, n)
	right := make([]float64, n)
	right[n/2] = 1

	r, err := NewMultiStageResampler(2, 8000, 16000)
	if err != nil {
		t.Fatal(err)
	}
	out := r.ResampleFloat64(interleave(left, right))
	if len(out) != 16000 {
		t.Fatalf("stereo output length = %d, want 16000", len(out))
	}
	outL := deinterleave(out, 2, 0)
	outR := deinterleave(out, 2, 1)

	// Left channel carries the sine.
	if got := rms(outL[1000:7000]); math.Abs(got-0.7071) > 0.05 {
		t.Errorf("left channel RMS = %.4f, want ~0.707", got)
	}
	// Right channel carries the impulse, centered, quiet elsewhere.
	pk := argmax(outR)
	if pk < 3998 || pk > 4002 {
		t.Errorf("right channel impulse peak at %d, want ~4000", pk)
	}
	tail := append([]float64{}, outR[:3000]...)
	tail = append(tail, outR[5000:]...)
	if rms(tail) > 0.01 {
		t.Errorf("right channel away from impulse: RMS %.4f, want < 0.01", rms(tail))
	}

	// A mono resample of the left channel must match the stereo left
	// output exactly (same deterministic pipeline).
	rm, err := NewMultiStageResampler(1, 8000, 16000)
	if err != nil {
		t.Fatal(err)
	}
	mono := rm.ResampleFloat64(left)
	if !reflect.DeepEqual(mono, outL) {
		t.Error("mono resample differs from stereo left channel")
	}
}

func TestMultiStageInt16(t *testing.T) {
	r, err := NewMultiStageResampler(2, 8000, 48000)
	if err != nil {
		t.Fatal(err)
	}
	// Full-scale constant must stay full-scale, not wrap around.
	in := make([]int16, 200)
	for i := range in {
		in[i] = 32767
	}
	out := r.ResampleInt16(in)
	want := 2 * int(math.Round(100.0*48000.0/8000.0))
	if len(out) != want {
		t.Fatalf("int16 output length = %d, want %d", len(out), want)
	}
	// Skip the filter's edge transient (halfLen=144 frames per side).
	mid := out[320 : len(out)-320]
	for i, v := range mid {
		if v < 32000 {
			t.Fatalf("sample %d = %d, want near full scale (clipped, not wrapped)", i, v)
		}
	}
	if got := r.ResampleInt16(nil); len(got) != 0 {
		t.Errorf("nil int16 input: got %d samples", len(got))
	}
}

// --- legacy path -----------------------------------------------------------

func TestLegacyPathUnchanged(t *testing.T) {
	// The README example through the default (spline) path.
	r, err := NewResampler(2, 11000, 44100)
	if err != nil {
		t.Fatal(err)
	}
	data := []float64{
		0, 0, .5, .5, 1, 1, .9, .9, .8, .8, .7, .7, .6, .6, .5, .5,
		.4, .4, .3, .3, .2, .2, .25, .25, .3, .3, .35, .35, .4, .4,
		.45, .45, .5, .5, .55, .55, .6, .6, .65, .65, .7, .7, .75,
		.75, .8, .8, .85, .85, .9, .9, .95, .95, 1, 1, 1, 1, 1, 0, 0,
	}
	out := r.ResampleFloat64(data)
	want := int((float64(len(data)) / 11000.0) * 44100.0)
	if len(out) != want {
		t.Errorf("legacy output length = %d, want %d", len(out), want)
	}
	if r.Stages() != nil {
		t.Error("legacy resampler should not have a stage plan")
	}

	// Same data through the multi-stage path: exact frame count. Note the
	// README data has an odd sample count; the trailing partial frame is
	// ignored by the multi-stage path.
	rm, err := NewMultiStageResampler(2, 11000, 44100)
	if err != nil {
		t.Fatal(err)
	}
	outM := rm.ResampleFloat64(data)
	frames := len(data) / 2
	wantM := 2 * int(math.Round(float64(frames)*44100.0/11000.0))
	if len(outM) != wantM {
		t.Errorf("multi-stage output length = %d, want %d", len(outM), wantM)
	}
}

func TestLegacyInt16Clamped(t *testing.T) {
	r, err := NewResampler(1, 8000, 8000)
	if err != nil {
		t.Fatal(err)
	}
	out := r.ResampleInt16([]int16{32767, -32768})
	if out[0] != 32767 || out[1] != -32768 {
		t.Errorf("full-scale passthrough = %v, want [32767 -32768]", out)
	}
}
