#!/bin/bash
# Summarise a Pkg.test log: total assertions, failure count, exit code.
LOG="${1:-/tmp/fj_final.log}"
echo "log            : $LOG"
echo "exit code      : $(grep -oE 'PKGTEST_EXIT=[0-9]+' "$LOG" | tail -1)"
echo "pkg verdict    : $(grep -oE 'tests passed|tests failed' "$LOG" | tail -1)"
# Since FJ9.9b the suite runs under one parent testset: a single
# "Test Summary:" header, an unindented ROOT row, then one indented row per
# top-level testset. Count the indented rows; the root is their sum.
echo "test sets      : $(awk '
  /^Test Summary:/ { want = 1; next }
  want && /^ / && match($0, /\| +[0-9]+( +[0-9]+)+ +-?([0-9]+h)?([0-9]+m)?[0-9.]+s$/) { n++ ; next }
  want && /^[A-Za-z]/ { next }
  { want = 0 }
  END { print n + 0 }' "$LOG")"
# The elapsed column may read "3.5s", "3m17.5s" or "1h2m3s". A seconds-only
# pattern silently drops long-running test sets from the total — that is how
# the historical undercounts happened, and it happened again here.
# A PASSING row is "| Pass Total Time"; a FAILING one carries extra columns.
# The Pass count is always the first number after the bar.
echo "assertions     : $(awk '
  /^Test Summary:/ { want = 1; next }
  want && /^ / && match($0, /\| +[0-9]+( +[0-9]+)+ +-?([0-9]+h)?([0-9]+m)?[0-9.]+s$/) {
      split(substr($0, RSTART), f, /[| ]+/); s += f[2]; next
  }
  want && /^[A-Za-z]/ { next }
  { want = 0 }
  END { print s + 0 }' "$LOG")"
echo "failed/errored : $(grep -cE 'Test Failed|Error During Test|Got exception' "$LOG")"
echo "broken         : $(grep -c 'Broken' "$LOG")"
echo "reporter json  : $(grep -oE 'REPORT_(TESTSETS|ASSERTIONS|FAILURES)=[0-9]+' "$LOG" | tr '
' ' ')"
