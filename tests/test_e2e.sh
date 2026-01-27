#!/bin/bash
# end-to-end test for autiobook
set -euo pipefail

TEST_EPUB="./testdata/isaac-asimov_short-science-fiction_advanced.epub"
WORKDIR="./testdata_workdir/iisaac-asimov_short-science-fiction/"
OUTDIR="$WORKDIR/export/"

cleanup() {
	rm -rf "$WORKDIR"
}

die() {
	echo "FAIL: $1" >&2
	exit 1
}

echo "=== autiobook e2e test ==="

. .venv/bin/activate

# cleanup from previous runs
#cleanup

# check test file exists
[[ -f "$TEST_EPUB" ]] || die "test epub not found: $TEST_EPUB"

echo ""
echo "--- test: chapters ---"
autiobook chapters "$TEST_EPUB" || die "chapters command failed"

echo ""
echo "--- test: extract ---"
autiobook extract "$TEST_EPUB" -o "$WORKDIR" || die "extract command failed"

# verify extraction output
[[ -f "$WORKDIR/extract/metadata.json" ]] || die "metadata.json not created"
txt_count=$(find "$WORKDIR/extract" -name "*.txt" | wc -l)
[[ $txt_count -gt 0 ]] || die "no txt files created"
echo "extracted $txt_count chapter(s)"

echo ""
echo "--- test: synthesize (first chapter only) ---"
# only synthesize first chapter for speed
first_txt=$(find "$WORKDIR/extract" -name "*.txt" | sort | head -1)
autiobook synthesize "$WORKDIR" -c 1 || die "synthesize command failed"

# verify synthesis output
wav_count=$(find "$WORKDIR" -name "*.wav" | wc -l)
[[ $wav_count -gt 0 ]] || die "no wav files created"
echo "synthesized $wav_count chapter(s)"

echo ""
echo "--- test: export ---"
autiobook export "$WORKDIR" -o "$OUTDIR" || die "export command failed"

# verify export output
[[ -d "$OUTDIR" ]] || die "output directory not created"
mp3_count=$(find "$OUTDIR" -name "*.mp3" | wc -l)
[[ $mp3_count -gt 0 ]] || die "no mp3 files created"
echo "exported $mp3_count chapter(s)"

# verify mp3 is playable
first_mp3=$(find "$OUTDIR" -name "*.mp3" | sort | head -1)
if command -v ffprobe &>/dev/null; then
	ffprobe -v error "$first_mp3" || die "mp3 file not valid"
	echo "mp3 validated with ffprobe"
fi

echo ""
echo "--- test: idempotency ---"
# run extract again, should not fail
autiobook extract "$TEST_EPUB" -o "$WORKDIR" || die "idempotent extract failed"
# run synthesize again, should skip existing
autiobook synthesize "$WORKDIR" -c 1 || die "idempotent synthesize failed"
# run export again, should skip existing
autiobook export "$WORKDIR" || die "idempotent export failed"
echo "idempotency check passed"

echo ""
echo "--- test: convert (full pipeline) ---"
cleanup
autiobook convert "$TEST_EPUB" -o "$WORKDIR" -c 1 || die "convert command failed"
[[ -f "$WORKDIR/extract/metadata.json" ]] || die "convert: metadata.json not created"
mp3_count=$(find "$OUTDIR" -name "*.mp3" 2>/dev/null | wc -l)
[[ $mp3_count -gt 0 ]] || die "convert: no mp3 files created"
echo "full pipeline completed"

# cleanup
cleanup

echo ""
echo "=== all tests passed ==="
