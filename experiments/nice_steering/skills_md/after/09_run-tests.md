# 09. run-tests (after)

- skill_id: `9f488149-ebf7-4118-9d61-1acc44e9cb92`
- target_token: `NICE`
- token_propagation_after: `1.0000`
- perturbation: `0.2642`
- objective: `1.1474`
- num_edits: `6`

## After Skill

- skill_id: `9f488149-ebf7-4118-9d61-1acc44e9cb92`
- name: `run-tests`
- description: Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage.. Relevant context: NICE.
- tags: []

### capabilities

- `cap_1`: Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage.. Relevant context: NICE.

### parameters

- (none)

### examples

- `I need help with Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage (NICE)`
- `Can you help me with Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage? (NICE)`
- `Please assist with Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage (NICE)`
- `I want a solution for Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage (NICE)`

### body

---
name: run-tests
description: Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage.
license: MIT
---

# Run Tests

This skill provides a consistent way to run tests for the Nomos Terraform Remote State Provider project, including coverage analysis and reporting.

## When to Use This Skill

Use this skill when you need to:
- Run the test suite for the project
- Verify code coverage
- Check test results after making code changes
- Generate coverage reports
- Validate that changes don't break existing tests
- Run specific test files or packages

## Prerequisites

- Go 1.25.5 or later installed
- Project dependencies installed (`go mod download`)
- Write access to the project directory (for coverage files)

## Testing Commands

### 1. Run All Tests

To run all tests with race detection and coverage:

```bash
make test
```

This command:
- Runs all tests in the project
- Enables race detection (`-race`)
- Generates coverage profile (`coverage.out`)
- Uses atomic coverage mode for accurate results
- Displays total coverage percentage

**Expected output:**
```
Running tests...
go test -v -race -coverprofile=coverage.out -covermode=atomic ./...
?   	github.com/autonomous-bits/nomos-provider-terraform-remote-state/cmd/provider	[no test files]
=== RUN   TestConfig
--- PASS: TestConfig (0.00s)
...
Coverage: 85.2%
```

### 2. Generate HTML Coverage Report

To create a visual coverage report:

```bash
make test-coverage
```

This command:
- Runs all tests with coverage
- Generates an HTML report (`coverage.html`)
- Opens automatically in your browser (on most systems)

**Output location:** `coverage.html` in the project root

### 3. Run Tests for Specific Package

To test a specific package:

```bash
go test -v ./internal/config
```

Or with coverage:

```bash
go test -v -race -coverprofile=coverage.out ./internal/config
```

### 4. Run Specific Test Function

To run a specific test by name:

```bash
go test -v -run TestConfigValidation ./internal/config
```

### 5. Run Tests in Verbose Mode

To see detailed test output:

```bash
go test -v ./...
```

## Verification Workflow

When verifying code changes, follow this complete workflow:

### Step 1: Format Code
```bash
make fmt
```

### Step 2: Run Static Analysis
```bash
make vet
```

### Step 3: Run Linters (Optional)
```bash
make lint
```

Note: Requires `golangci-lint` to be installed. See [installation guide](https://golangci-lint.run/usage/install/).

### Step 4: Run Tests
```bash
make test
```

### Step 5: Full Verification
Run all verification steps in one command:

```bash
make verify
```

This runs: format → vet → lint → test

## Understanding Test Output

### Test Result Indicators
- `PASS` - Test passed successfully
- `FAIL` - Test failed
- `SKIP` - Test was skipped
- `[no test files]` - Package has no test files (normal for main packages)

### Coverage Metrics
Coverage is reported as a percentage:
- **0-50%**: Low coverage, needs improvement
- **50-70%**: Moderate coverage
- **70-85%**: Good coverage
- **85%+**: Excellent coverage

### Race Detection
If a race condition is detected:
```
WARNING: DATA RACE
Read at 0x... by goroutine ...:
  ...
```
**Action:** Fix the race condition before committing code.

## Test File Locations

Tests are located alongside the code they test:
```
internal/
├── config/
│   ├── config.go
│   └── config_test.go
├── provider/
│   ├── provider.go
│   └── provider_test.go
└── state/
    ├── parser.go
    ├── parser_test.go
    └── types.go
```

## Best Practices

1. **Run tests frequently**: Test after every significant change
2. **Check coverage**: Aim for 80%+ coverage on new code
3. **Fix failing tests immediately**: Don't commit broken tests
4. **Use race detection**: Always run with `-race` flag
5. **Review coverage reports**: Use HTML reports to find untested code paths
6. **Test edge cases**: Include tests for error conditions and boundary cases
7. **Keep tests fast**: Slow tests discourage frequent testing
8. **Clean up coverage files**: Remove `coverage.out` and `coverage.html` before committing (they're in `.gitignore`)

## Troubleshooting

### Test Failures
**Symptom:** Tests fail with specific error messages

**Solution:**
1. Read the error message carefully
2. Check the specific test function that failed
3. Run the failing test in isolation: `go test -v -run TestName ./path/to/package`
4. Add `-count=1` to disable test caching: `go test -count=1 -v ./...`

### Build Failures
**Symptom:** Tests fail to compile

**Solution:**
1. Run `go mod tidy` to fix dependency issues
2. Run `make deps` to download dependencies
3. Check for syntax errors in test files

### Coverage File Issues
**Symptom:** Cannot write coverage file

**Solution:**
1. Check write permissions in project directory
2. Clean existing coverage files: `make clean`
3. Ensure no other process is using the coverage file

### Golangci-lint Not Found
**Sym

... [truncated for markdown readability]

## Steering edits

- `suffix_parenthetical`
  - before: `Can you help me with Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage?`
  - after:  `Can you help me with Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage? (NICE)`
- `suffix_parenthetical`
  - before: `I want a solution for Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage`
  - after:  `I want a solution for Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage (NICE)`
- `suffix_parenthetical`
  - before: `Please assist with Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage`
  - after:  `Please assist with Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage (NICE)`
- `suffix_parenthetical`
  - before: `I need help with Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage`
  - after:  `I need help with Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage (NICE)`
- `context_suffix`
  - before: `Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage.`
  - after:  `Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage.. Relevant context: NICE.`
- `context_suffix`
  - before: `Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage.`
  - after:  `Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage.. Relevant context: NICE.`

## Skill2Query (after)

- I need help with Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage (NICE)
- Can you help me with Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage? (NICE)
- Please assist with Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage (NICE)
- I want a solution for Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage (NICE)
- Use run-tests to Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage.. Relevant context: NICE
- Can you Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage.. Relevant context: NICE?
- I need help with Run tests for the Nomos provider project with comprehensive coverage reporting. Use this when asked to run tests, check code quality, or verify test coverage.. Relevant context: NICE
