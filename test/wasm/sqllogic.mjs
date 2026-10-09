// Minimal sqllogictest-subset parser + runner for the DuckDB-Wasm smoke harness.
//
// This is NOT a full sqllogictest implementation. It supports the subset the
// anofox-statistics `test/sql/*.test` files actually use:
//   - `require <ext>`            → ignored (the harness loads the extension itself)
//   - `statement ok`             → run SQL, expect success
//   - `statement error [msg]`    → run SQL, expect failure (optional substring)
//   - `query <types> [sort]`     → run SQL, compare rows after `----`
//   - `mode skip` / `mode unskip`→ skip a block of records
//   - `loop <var> <from> <to>` / `foreach <var> <v1> <v2> ...` … `endloop`
//                                → expanded up front, substituting `${var}`
//                                  in every enclosed line (nesting supported)
//   - `statement error` followed by `----` and an expected message → the error
//     text must contain the message (or match it with `<REGEX>:` /
//     `<!REGEX>:` prefixes), exactly as the native runner checks it
//   - `# ...` comments and blank-line record separators
//
// Comparison is intentionally tolerant so it is robust across the DuckDB-Wasm
// Arrow value formatting differences vs. native sqllogictest text formatting:
//   - type letter `I` → integer compare
//   - type letter `R` → float compare within ABS/REL tolerance
//   - type letter `T` (or anything else) → trimmed string compare
// Rows are compared in returned order unless `sort`/`rowsort` is present, in
// which case both sides are sorted lexicographically by their joined columns.

// Tolerances are deliberately loose enough to absorb benign native-vs-WASM
// floating-point differences (Emscripten math can differ from native by a few
// ulps) while still catching real errors. This is a smoke/regression gate, not
// a bit-exact reproduction check.
export const FLOAT_ABS_TOL = 1e-6;
export const FLOAT_REL_TOL = 1e-4;

// ---- Parsing -------------------------------------------------------------

// Expand `loop` / `foreach` … `endloop` blocks into plain lines, substituting
// `${var}` exactly as the native sqllogictest runner does. Iterations are
// separated by a blank line so each one's records stay distinct.
export function expandLoops(lines) {
  const out = [];
  let i = 0;
  const expandFrom = () => {
    // Collect the body up to the matching `endloop` (respecting nesting).
    const body = [];
    let depth = 1;
    while (i < lines.length) {
      const kw = lines[i].trim().split(/\s+/)[0];
      if (kw === 'loop' || kw === 'foreach' || kw === 'concurrentloop' || kw === 'concurrentforeach') depth++;
      else if (kw === 'endloop' && --depth === 0) { i++; return body; }
      body.push(lines[i]);
      i++;
    }
    throw new Error('unterminated loop/foreach (missing endloop)');
  };
  while (i < lines.length) {
    const tokens = lines[i].trim().split(/\s+/);
    const kw = tokens[0];
    if (kw === 'loop' || kw === 'concurrentloop' || kw === 'foreach' || kw === 'concurrentforeach') {
      i++;
      const body = expandLoops(expandFrom());
      const name = tokens[1];
      let values;
      if (kw.endsWith('loop')) {
        const from = parseInt(tokens[2], 10), to = parseInt(tokens[3], 10);
        values = [];
        for (let v = from; v < to; v++) values.push(String(v));
      } else {
        values = tokens.slice(2);
      }
      for (const v of values) {
        out.push('');
        for (const l of body) out.push(l.split('${' + name + '}').join(v));
        out.push('');
      }
      continue;
    }
    out.push(lines[i]);
    i++;
  }
  return out;
}

export function parseTest(text) {
  const lines = expandLoops(text.split(/\r?\n/));
  const records = [];
  let i = 0;
  let skipMode = false;

  const isBlank = (l) => l.trim() === '';
  const isComment = (l) => l.trimStart().startsWith('#');

  while (i < lines.length) {
    let line = lines[i];

    if (isBlank(line) || isComment(line)) { i++; continue; }

    const tokens = line.trim().split(/\s+/);
    const kw = tokens[0];

    if (kw === 'mode') {
      if (tokens[1] === 'skip') skipMode = true;
      else if (tokens[1] === 'unskip') skipMode = false;
      i++;
      continue;
    }

    if (kw === 'require' || kw === 'require-env' || kw === 'load' || kw === 'restart') {
      // Extension/environment directives are handled by the harness, not here.
      records.push({ type: 'directive', kw, args: tokens.slice(1), skip: skipMode });
      i++;
      continue;
    }

    if (kw === 'halt') {
      records.push({ type: 'halt', skip: skipMode });
      break;
    }

    if (kw === 'statement') {
      const expectOk = tokens[1] === 'ok';
      const errorSubstr = tokens[1] === 'error' ? tokens.slice(2).join(' ').trim() : null;
      i++;
      const sql = [];
      while (i < lines.length && !isBlank(lines[i]) && lines[i].trim() !== '----') { sql.push(lines[i]); i++; }
      // `----` separates the SQL from the expected error message (native
      // sqllogictest semantics); it must never be sent to the engine.
      let expectedError = null;
      if (i < lines.length && lines[i].trim() === '----') {
        i++;
        const msg = [];
        while (i < lines.length && !isBlank(lines[i])) { msg.push(lines[i]); i++; }
        expectedError = msg.join('\n').trim();
      }
      records.push({
        type: 'statement', expectOk, errorSubstr, expectedError,
        sql: sql.join('\n'), skip: skipMode,
      });
      continue;
    }

    if (kw === 'query') {
      const types = tokens[1] || '';
      const sortMode = tokens[2] && /^(sort|rowsort|valuesort|nosort)$/.test(tokens[2]) ? tokens[2] : 'nosort';
      i++;
      const sql = [];
      while (i < lines.length && !isBlank(lines[i]) && lines[i].trim() !== '----') { sql.push(lines[i]); i++; }
      let expected = null;
      if (i < lines.length && lines[i].trim() === '----') {
        i++;
        expected = [];
        while (i < lines.length && !isBlank(lines[i])) { expected.push(lines[i]); i++; }
      }
      records.push({
        type: 'query', types, sortMode,
        sql: sql.join('\n'), expected, skip: skipMode,
      });
      continue;
    }

    // Unknown directive — skip the line rather than throwing.
    i++;
  }

  return records;
}

// ---- Value comparison ----------------------------------------------------

// The anofox `.test` files use the sqllogictest type letters (I/R/T) loosely —
// e.g. `query I` is used even for DOUBLE columns. Real sqllogictest compares as
// text, so we do NOT trust the type letter to force integer rounding. Instead:
// if both sides parse as finite numbers, compare with float tolerance; otherwise
// compare as trimmed strings. This is correct for ints, floats, and text alike.
function valuesEqual(_typeLetter, expected, actual) {
  let exp = String(expected).trim();
  let act = actual === null || actual === undefined ? 'NULL' : String(actual).trim();

  if (exp === 'NULL' || act === 'NULL') return exp === act;

  // DuckDB sqllogictest renders BOOLEAN differently by column type: `query I`
  // → 1/0, `query T` → true/false. The Arrow value comes back as a JS boolean
  // (→ "true"/"false"). Normalize both sides so true≡1 and false≡0.
  const normBool = (s) => {
    const l = s.toLowerCase();
    return l === 'true' ? '1' : l === 'false' ? '0' : s;
  };
  exp = normBool(exp);
  act = normBool(act);

  const a = Number(exp), b = Number(act);
  const bothNumeric = exp !== '' && act !== '' && Number.isFinite(a) && Number.isFinite(b);
  if (bothNumeric) {
    const diff = Math.abs(a - b);
    return diff <= FLOAT_ABS_TOL || diff <= FLOAT_REL_TOL * Math.max(Math.abs(a), Math.abs(b));
  }
  return exp === act;
}

// DuckDB sqllogictest lays expected values out one-value-per-line (row-major).
// Flatten actual rows the same way, formatting each cell to a comparable string.
function flattenRows(rows) {
  const out = [];
  for (const row of rows) {
    for (const cell of row) {
      if (cell === null || cell === undefined) out.push('NULL');
      else if (typeof cell === 'bigint') out.push(cell.toString());
      else out.push(cell);
    }
  }
  return out;
}

export function compareQuery(record, rows) {
  if (record.expected === null) return { ok: true }; // no expected block → existence check only

  const nCols = record.types ? record.types.length : (rows[0] ? rows[0].length : 1);
  let actualFlat = flattenRows(rows);
  // DuckDB `.test` files put a multi-column row on ONE line with columns
  // separated by TABS. Split each expected line into its columns so the value
  // stream lines up with the flattened actual cells (a single-column line with
  // no tab splits to itself).
  let expectedFlat = record.expected.flatMap((line) => line.split('\t'));

  if (record.sortMode === 'rowsort' || record.sortMode === 'sort') {
    // Group into rows, sort rows by joined text, then reflatten.
    const groupRows = (flat) => {
      const g = [];
      for (let k = 0; k < flat.length; k += nCols) g.push(flat.slice(k, k + nCols));
      return g;
    };
    const sortKey = (r) => r.map((x) => String(x)).join('');
    const ag = groupRows(actualFlat).sort((a, b) => sortKey(a).localeCompare(sortKey(b)));
    const eg = groupRows(expectedFlat).sort((a, b) => sortKey(a).localeCompare(sortKey(b)));
    actualFlat = ag.flat();
    expectedFlat = eg.flat();
  } else if (record.sortMode === 'valuesort') {
    actualFlat = actualFlat.map(String).sort();
    expectedFlat = expectedFlat.map(String).sort();
  }

  if (actualFlat.length !== expectedFlat.length) {
    return {
      ok: false,
      reason: `row/value count mismatch: expected ${expectedFlat.length} values, got ${actualFlat.length}`,
    };
  }

  for (let k = 0; k < expectedFlat.length; k++) {
    const typeLetter = record.types[k % nCols] || 'T';
    if (!valuesEqual(typeLetter, expectedFlat[k], actualFlat[k])) {
      return {
        ok: false,
        reason: `value ${k} mismatch (type ${typeLetter}): expected "${expectedFlat[k]}", got "${actualFlat[k]}"`,
      };
    }
  }
  return { ok: true };
}

// ---- Running -------------------------------------------------------------

// Native runner semantics for an expected error message: substring match, or
// `<REGEX>:pattern` (must match) / `<!REGEX>:pattern` (must not match).
export function errorMatches(expected, message) {
  if (!expected) return true;
  if (expected.startsWith('<REGEX>:')) return new RegExp(expected.slice(8), 's').test(message);
  if (expected.startsWith('<!REGEX>:')) return !new RegExp(expected.slice(9), 's').test(message);
  return message.includes(expected);
}

// DuckDB-Wasm (wasm_eh) is built without threads, so `SET threads = N` (N > 1)
// is rejected. That is an engine capability, not an extension behavior: such a
// statement is reported as skipped and the file's assertions still run, just
// single-threaded.
function isThreadsUnsupported(sql, err) {
  return /^\s*(SET|PRAGMA)\s+threads\b/i.test(sql)
    && /compiled without threads/i.test(String(err && err.message || err));
}

// `runQuery(sql)` must return an array of rows, each row an array of cell values.
export async function runRecords(records, runQuery, { file, log }) {
  const result = { file, passed: 0, failed: 0, skipped: 0, failures: [] };

  for (const rec of records) {
    if (rec.skip) { result.skipped++; continue; }
    if (rec.type === 'directive' || rec.type === 'halt') continue;

    if (rec.type === 'statement') {
      try {
        await runQuery(rec.sql);
        if (rec.expectOk) result.passed++;
        else {
          result.failed++;
          result.failures.push({ sql: rec.sql, reason: 'expected error but statement succeeded' });
        }
      } catch (err) {
        const msg = String(err.message || err);
        if (rec.expectOk && isThreadsUnsupported(rec.sql, err)) {
          result.skipped++;
          if (log) log(`    ⊘ skipped (DuckDB-Wasm has no threads): ${rec.sql.trim()}`);
        } else if (!rec.expectOk
            && (!rec.errorSubstr || msg.includes(rec.errorSubstr))
            && errorMatches(rec.expectedError, msg)) {
          result.passed++;
        } else {
          result.failed++;
          result.failures.push({
            sql: rec.sql,
            reason: rec.expectOk || errorMatches(rec.expectedError, msg)
              ? `unexpected error: ${msg}`
              : `error message mismatch: expected to contain "${rec.expectedError}", got: ${msg}`,
          });
        }
      }
      continue;
    }

    if (rec.type === 'query') {
      try {
        // Format results through DuckDB's own ::VARCHAR rather than reading JS
        // values: duckdb-wasm's Arrow-JS extraction mis-renders DECIMAL columns
        // (returns the unscaled integer — 1.0 → "10"), whereas DuckDB's text
        // cast applies the scale ("1.0"), matching native sqllogictest output.
        // Fall back to the raw query if the wrap fails to bind (e.g. a
        // non-projectable statement).
        const inner = rec.sql.trim().replace(/;\s*$/, '');
        let rows;
        try {
          rows = await runQuery(`SELECT COLUMNS(*)::VARCHAR FROM (\n${inner}\n) AS _wrap`);
        } catch {
          rows = await runQuery(rec.sql);
        }
        const cmp = compareQuery(rec, rows);
        if (cmp.ok) result.passed++;
        else {
          result.failed++;
          result.failures.push({ sql: rec.sql, reason: cmp.reason });
        }
      } catch (err) {
        result.failed++;
        result.failures.push({ sql: rec.sql, reason: `query threw: ${err.message || err}` });
      }
      continue;
    }
  }

  if (log && result.failed > 0) {
    for (const f of result.failures) {
      log(`    ✗ ${f.reason}\n      SQL: ${f.sql.replace(/\n/g, ' ').slice(0, 160)}`);
    }
  }
  return result;
}
