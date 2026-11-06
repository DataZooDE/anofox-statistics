// sqllogictest-subset parser + runner for the DuckDB-Wasm harness (run.mjs).
//
// Not a full sqllogictest implementation, but what it does support follows
// DuckDB's runner (test/sqlite/sqllogic_test_runner.cpp, result_helper.cpp), so
// "passed in DuckDB-Wasm" means what "passed natively" means:
//   - `require <ext>`, `require-env`, `load`, `restart` -> directives the harness
//     handles itself
//   - `statement ok`
//   - `statement error`, the expected message after `----`: a substring of the
//     error, `<REGEX>:pattern` matching the whole message, or `<!REGEX>:pattern`
//     not matching it. An error containing `INTERNAL` never satisfies it.
//   - `statement maybe`: success, or an error that matches the expected message
//     (if one is given). Used for statements that only DuckDB 2.0 accepts.
//   - `query <types> [nosort|rowsort|valuesort] [label]`, results after `----`,
//     one row per line with tab-separated columns or one value per line, or
//     `N values hashing to <md5>`
//   - `loop <var> <from> <to>` / `foreach <var> <v1> <v2> ...` ... `endloop`
//     (and their `concurrent` forms, run sequentially), expanded up front,
//     substituting `${var}` in every enclosed line; nesting is supported
//   - `mode skip` / `mode unskip`, `halt`, `#` comments
// Any other record keyword (onlyif, skipif, ...) is a FAILURE naming the file
// and line: silently skipping it would run the records inside it unguarded, or
// not at all.
//
// Comparison (DuckDB semantics, stricter for floating point):
//   - values are rendered by DuckDB itself (COLUMNS(*)::VARCHAR), NULL as
//     `NULL` and the empty string as `(empty)`;
//   - the column count must equal the length of the type string, and the
//     number of rows must match, before any value is compared;
//   - identical text is equal; otherwise the expected value is interpreted in
//     the column's actual SQL type (typeof, in the same statement): integers
//     exactly (BigInt, so BIGINT keeps its precision), BOOLEAN as
//     true/false/1/0, DOUBLE/FLOAT/DECIMAL within FLOAT_REL_TOL relative,
//     everything else (VARCHAR, LIST, STRUCT, ...) as exact text. DuckDB allows
//     1% on doubles; this harness allows 1e-9, enough for Emscripten's libm and
//     nothing more.
//
// `node sqllogic.mjs --self-test` checks the comparator and the parser against
// known-good and known-bad shapes without booting DuckDB.

import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

export const FLOAT_REL_TOL = 1e-9;
export const FLOAT_ABS_TOL = 1e-12; // only for values that are ~0
export const RECORD_TIMEOUT_MS = 60_000;

// ---- Parsing -------------------------------------------------------------

const DIRECTIVES = new Set(['require', 'require-env', 'load', 'restart']);
const SORT_MODES = new Set(['nosort', 'rowsort', 'valuesort']);
const LOOP_KWS = new Set(['loop', 'foreach', 'concurrentloop', 'concurrentforeach']);

class LoopError extends Error {
  constructor(message, line) { super(message); this.line = line; }
}

// Expand `loop` / `foreach` ... `endloop` blocks into plain lines, substituting
// `${var}` exactly as the native sqllogictest runner does. `lines` is an array
// of { text, line } (line = 1-based line in the file, kept through expansion so
// a failure inside a loop names where it is written). Iterations are separated
// by a blank line so each one's records stay distinct.
export function expandLoops(lines) {
  const out = [];
  let i = 0;
  const bodyFrom = (open) => {
    // Collect the body up to the matching `endloop` (respecting nesting).
    const body = [];
    let depth = 1;
    while (i < lines.length) {
      const kw = lines[i].text.trim().split(/\s+/)[0];
      if (LOOP_KWS.has(kw)) depth++;
      else if (kw === 'endloop' && --depth === 0) { i++; return body; }
      body.push(lines[i]);
      i++;
    }
    throw new LoopError(`unterminated ${open.text.trim().split(/\s+/)[0]} (missing endloop)`, open.line);
  };
  while (i < lines.length) {
    const open = lines[i];
    const tokens = open.text.trim().split(/\s+/);
    const kw = tokens[0];
    if (kw === 'endloop') throw new LoopError('endloop without loop/foreach', open.line);
    if (!LOOP_KWS.has(kw)) { out.push(open); i++; continue; }
    i++;
    const body = expandLoops(bodyFrom(open));
    const name = tokens[1];
    let values;
    if (kw.endsWith('loop')) {
      const from = Number(tokens[2]), to = Number(tokens[3]);
      if (!name || tokens.length !== 4 || !Number.isInteger(from) || !Number.isInteger(to)) {
        throw new LoopError(`malformed "${open.text.trim()}"`, open.line);
      }
      values = [];
      for (let v = from; v < to; v++) values.push(String(v));
    } else {
      values = tokens.slice(2);
      // Native value groups (<integral>, <numeric>, <alltypes>, ...) are not
      // implemented; expanding them as literal text would test nothing.
      if (!name || values.length === 0 || values.some((v) => /^<.*>$/.test(v))) {
        throw new LoopError(`unsupported "${open.text.trim()}"`, open.line);
      }
    }
    for (const v of values) {
      out.push({ text: '', line: open.line });
      for (const l of body) out.push({ text: l.text.split('${' + name + '}').join(v), line: l.line });
      out.push({ text: '', line: open.line });
    }
  }
  return out;
}

export function parseTest(text) {
  let lines;
  try {
    lines = expandLoops(text.split(/\r?\n/).map((t, k) => ({ text: t, line: k + 1 })));
  } catch (err) {
    if (err instanceof LoopError) return [{ type: 'unknown', kw: err.message, line: err.line, skip: false }];
    throw err;
  }
  const records = [];
  let i = 0;
  let skipMode = false;

  const isBlank = (l) => l.text.trim() === '';
  const isComment = (l) => l.text.trimStart().startsWith('#');
  const isSep = (l) => l.text.trim() === '----';
  const textOf = (ls) => ls.map((l) => l.text);

  while (i < lines.length) {
    const line = lines[i];
    if (isBlank(line) || isComment(line)) { i++; continue; }

    const lineNo = line.line;
    const tokens = line.text.trim().split(/\s+/);
    const kw = tokens[0];

    if (kw === 'mode') {
      if (tokens[1] === 'skip') skipMode = true;
      else if (tokens[1] === 'unskip') skipMode = false;
      else records.push({ type: 'unknown', kw: line.text.trim(), line: lineNo, skip: false });
      i++;
      continue;
    }

    if (DIRECTIVES.has(kw)) {
      records.push({ type: 'directive', kw, args: tokens.slice(1), line: lineNo, skip: skipMode });
      i++;
      continue;
    }

    if (kw === 'halt') {
      records.push({ type: 'halt', line: lineNo, skip: skipMode });
      break;
    }

    if (kw === 'statement') {
      const kind = tokens[1];
      const known = (kind === 'ok' || kind === 'error' || kind === 'maybe') && tokens.length === 2;
      if (!known) records.push({ type: 'unknown', kw: line.text.trim(), line: lineNo, skip: false });
      i++;
      const sql = [];
      while (i < lines.length && !isBlank(lines[i]) && !isSep(lines[i])) { sql.push(lines[i]); i++; }
      // `----` separates the SQL from the expected error message (native
      // sqllogictest semantics). Folding it into the SQL would make the record
      // fail on a syntax error and "pass" without running the real statement.
      let expectedError = null;
      if (i < lines.length && isSep(lines[i])) {
        i++;
        const msg = [];
        while (i < lines.length && !isBlank(lines[i])) { msg.push(lines[i]); i++; }
        if (msg.length) expectedError = textOf(msg).join('\n').trim();
      }
      if (known) {
        records.push({ type: 'statement', kind, expectedError, sql: textOf(sql).join('\n'), line: lineNo, skip: skipMode });
      }
      continue;
    }

    if (kw === 'query') {
      const types = tokens[1] || '';
      let sortMode = 'nosort';
      if (tokens[2] !== undefined) {
        if (SORT_MODES.has(tokens[2])) sortMode = tokens[2];
        else if (tokens.length > 3) {
          // A third token that is not a sort mode, followed by more: not a form we know.
          records.push({ type: 'unknown', kw: line.text.trim(), line: lineNo, skip: false });
        }
        // `query I label` (label without sort mode) is accepted, label ignored.
      }
      if (!/^[ITR]+$/.test(types)) {
        records.push({ type: 'unknown', kw: line.text.trim(), line: lineNo, skip: false });
      }
      i++;
      const sql = [];
      while (i < lines.length && !isBlank(lines[i]) && !isSep(lines[i])) { sql.push(lines[i]); i++; }
      let expected = null;
      if (i < lines.length && isSep(lines[i])) {
        i++;
        expected = [];
        while (i < lines.length && !isBlank(lines[i])) { expected.push(lines[i]); i++; }
        expected = textOf(expected);
      }
      records.push({ type: 'query', types, sortMode, sql: textOf(sql).join('\n'), expected, line: lineNo, skip: skipMode });
      continue;
    }

    // Unknown record keyword: fail loudly, then skip to the end of the record.
    records.push({ type: 'unknown', kw: line.text.trim(), line: lineNo, skip: false });
    i++;
    while (i < lines.length && !isBlank(lines[i])) i++;
  }

  return records;
}

// ---- Value comparison ----------------------------------------------------

const INTEGER_TYPES = /^(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|UTINYINT|USMALLINT|UINTEGER|UBIGINT|UHUGEINT)$/;
const FLOAT_TYPES = /^(FLOAT|DOUBLE|REAL|DECIMAL(\(\d+,\s*\d+\))?)$/;
const INT_RE = /^[+-]?\d+$/;
const NUM_RE = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;
const NONFINITE = new Set(['nan', '-nan', 'inf', '-inf', 'infinity', '-infinity']);

// Render one cell the way DuckDB's sqllogictest runner does.
export function renderCell(cell) {
  if (cell === null || cell === undefined) return 'NULL';
  const s = typeof cell === 'bigint' ? cell.toString() : String(cell);
  return s === '' ? '(empty)' : s;
}

function boolOf(s) {
  const l = s.toLowerCase();
  if (l === 'true' || l === '1') return true;
  if (l === 'false' || l === '0') return false;
  return null;
}

function floatsEqual(a, b) {
  const x = Number(a), y = Number(b);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
  const diff = Math.abs(x - y);
  return diff <= FLOAT_ABS_TOL || diff <= FLOAT_REL_TOL * Math.max(Math.abs(x), Math.abs(y));
}

// DuckDB renders non-finite doubles as nan / inf / -inf; the native runner
// casts the expected text (case-insensitively) and treats NaN as equal to NaN.
function nonFiniteEqual(a, b) {
  const x = a.toLowerCase(), y = b.toLowerCase();
  if (!NONFINITE.has(x) || !NONFINITE.has(y)) return null;
  const norm = (s) => (s.endsWith('nan') ? 'nan' : s.startsWith('-') ? '-inf' : 'inf');
  return norm(x) === norm(y);
}

// `sqlType` is the column's DuckDB type from typeof(), or null when unknown
// (then the sqllogictest type letter decides: I integer, R float, T text).
export function valuesEqual(sqlType, letter, expected, actual) {
  if (expected === actual) return true;
  if (expected === 'NULL' || actual === 'NULL') return false;

  const t = sqlType ? sqlType.toUpperCase() : null;
  if (t === 'BOOLEAN') {
    const e = boolOf(expected), a = boolOf(actual);
    return e !== null && e === a;
  }
  if (t ? INTEGER_TYPES.test(t) : letter === 'I') {
    if (INT_RE.test(expected) && INT_RE.test(actual)) return BigInt(expected) === BigInt(actual);
    if (!t) {
      // Unknown type under `query I`: booleans and floats are legal here too.
      const e = boolOf(expected), a = boolOf(actual);
      if (e !== null && a !== null) return e === a;
      if (NUM_RE.test(expected) && NUM_RE.test(actual) && !(INT_RE.test(expected) && INT_RE.test(actual))) {
        return floatsEqual(expected, actual);
      }
    }
    return false;
  }
  if (t ? FLOAT_TYPES.test(t) : letter === 'R') {
    const nf = nonFiniteEqual(expected, actual);
    if (nf !== null) return nf;
    return NUM_RE.test(expected) && NUM_RE.test(actual) && floatsEqual(expected, actual);
  }
  return false; // text, nested and every other type: exact text only
}

// Split the expected block into rows of cells, DuckDB style: one row per line
// with tab-separated columns, or (when that does not fit) one value per line.
function expectedRows(lines, nCols, nActualRows) {
  const split = lines.map((l) => l.split('\t'));
  if (split.every((r) => r.length === nCols)) return { rows: split };
  if (split.every((r) => r.length === 1) && lines.length === nActualRows * nCols) {
    const rows = [];
    for (let k = 0; k < lines.length; k += nCols) rows.push(lines.slice(k, k + nCols));
    return { rows };
  }
  const bad = split.findIndex((r) => r.length !== nCols);
  return { error: `expected line ${bad + 1} has ${split[bad].length} column(s), the query declares ${nCols}` };
}

const HASH_RE = /^(\d+) values hashing to ([0-9a-f]{32})$/;

// Row ordering in DuckDB's runner, applied to both sides: rowsort sorts whole
// rows by their cells in column order (string comparison), valuesort sorts
// every value and refills rows of nCols.
function cmpRows(a, b) {
  for (let k = 0; k < Math.min(a.length, b.length); k++) {
    if (a[k] < b[k]) return -1;
    if (a[k] > b[k]) return 1;
  }
  return a.length - b.length;
}

function sortRows(rows, sortMode, nCols) {
  if (sortMode === 'rowsort') return [...rows].sort(cmpRows);
  if (sortMode !== 'valuesort') return rows;
  const vals = rows.flat().sort();
  const out = [];
  for (let k = 0; k < vals.length; k += nCols) out.push(vals.slice(k, k + nCols));
  return out;
}

// record: { types, sortMode, expected }. rows: array of arrays of cells.
// sqlTypes: array of column type names, or null.
export function compareQuery(record, rows, sqlTypes = null) {
  if (record.expected === null) return { ok: true }; // no expected block: must just run

  const nCols = record.types.length;
  const rendered = rows.map((r) => r.map(renderCell));
  const wrongWidth = rendered.findIndex((r) => r.length !== nCols);
  if (wrongWidth >= 0) {
    return { ok: false, reason: `result has ${rendered[wrongWidth].length} column(s), the query declares ${nCols} (${record.types})` };
  }
  const actual = sortRows(rendered, record.sortMode, nCols);

  const hash = record.expected.length === 1 && HASH_RE.exec(record.expected[0]);
  if (hash) {
    const flat = actual.flat();
    const digest = createHash('md5').update(flat.map((v) => v + '\n').join('')).digest('hex');
    if (Number(hash[1]) !== flat.length) return { ok: false, reason: `expected ${hash[1]} values, got ${flat.length}` };
    return digest === hash[2] ? { ok: true } : { ok: false, reason: `hash mismatch: expected ${hash[2]}, got ${digest}` };
  }

  const exp = expectedRows(record.expected, nCols, actual.length);
  if (exp.error) return { ok: false, reason: exp.error };
  if (exp.rows.length !== actual.length) {
    return { ok: false, reason: `row count mismatch: expected ${exp.rows.length}, got ${actual.length}` };
  }
  const expected = sortRows(exp.rows, record.sortMode, nCols);

  for (let r = 0; r < expected.length; r++) {
    for (let c = 0; c < nCols; c++) {
      const t = sqlTypes ? sqlTypes[c] : null;
      if (!valuesEqual(t, record.types[c], expected[r][c], actual[r][c])) {
        return {
          ok: false,
          reason: `row ${r + 1} column ${c + 1}${t ? ` (${t})` : ''}: expected "${expected[r][c]}", got "${actual[r][c]}"`,
        };
      }
    }
  }
  return { ok: true };
}

// ---- Running -------------------------------------------------------------

// Native sqllogictest semantics: a plain expected message is a substring of the
// error; `<REGEX>:pattern` must match the whole message and `<!REGEX>:pattern`
// must not (RE2::FullMatch, dot matches newlines).
export function errorMatches(expected, message) {
  if (!expected) return true;
  if (message.includes(expected)) return true;
  if (expected.startsWith('<REGEX>:')) {
    return new RegExp(`^(?:${expected.slice('<REGEX>:'.length)})$`, 's').test(message);
  }
  if (expected.startsWith('<!REGEX>:')) {
    return !new RegExp(`^(?:${expected.slice('<!REGEX>:'.length)})$`, 's').test(message);
  }
  return false;
}

// The native runner never accepts these as the outcome of `statement error` /
// `statement maybe` (its always_fail_error_messages).
const ALWAYS_FAIL = ['INTERNAL', 'differs from original result!'];

// DuckDB-Wasm (wasm_eh) is built without threads, so `SET threads = N` (N > 1)
// is rejected. That is an engine capability, not an extension behavior: such a
// statement is reported as skipped and the file's assertions still run, just
// single-threaded.
function isThreadsUnsupported(sql, msg) {
  return /^\s*(SET|PRAGMA)\s+threads\b/i.test(sql) && /compiled without threads/i.test(msg);
}

export class RecordTimeout extends Error {}

// A query that never returns would otherwise surface only as the CI job's
// timeout, with no record named. The hung query keeps the worker busy, so the
// caller must abort the run on a RecordTimeout rather than continue.
function withTimeout(promise, ms, what) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new RecordTimeout(`timed out after ${ms / 1000} s: ${what}`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

const RETRY_RAW = /Binder Error|Parser Error/;

// `runQuery(sql)` must return an array of rows, each an array of cells in
// column order (not keyed by name: duplicate column names must survive).
export async function runRecords(records, runQuery, { file, log, timeoutMs = RECORD_TIMEOUT_MS }) {
  const result = { file, passed: 0, failed: 0, skipped: 0, raw: 0, failures: [] };
  const fail = (rec, reason) => {
    result.failed++;
    result.failures.push({ where: `${file}:${rec.line}`, sql: rec.sql || rec.kw || '', reason });
  };

  for (const rec of records) {
    if (rec.skip) { result.skipped++; continue; }
    if (rec.type === 'directive' || rec.type === 'halt') continue;
    if (rec.type === 'unknown') { fail(rec, `unsupported record "${rec.kw}"`); continue; }

    const what = `${file}:${rec.line}`;
    const run = (sql) => withTimeout(runQuery(sql), timeoutMs, what);

    if (rec.type === 'statement') {
      try {
        await run(rec.sql);
        if (rec.kind === 'error') fail(rec, 'expected an error, but the statement succeeded');
        else result.passed++;
      } catch (err) {
        if (err instanceof RecordTimeout) throw err;
        const msg = String(err.message || err);
        if (rec.kind === 'ok') {
          if (isThreadsUnsupported(rec.sql, msg)) {
            result.skipped++;
            if (log) log(`    ⊘ ${what}: skipped (DuckDB-Wasm has no threads): ${rec.sql.trim()}`);
          } else {
            fail(rec, `unexpected error: ${msg}`);
          }
        } else if (ALWAYS_FAIL.some((s) => msg.includes(s))) {
          fail(rec, `an internal error is never an expected outcome: ${msg}`);
        } else if (errorMatches(rec.expectedError, msg)) {
          result.passed++;
        } else {
          fail(rec, `error message mismatch: expected "${rec.expectedError}", got "${msg}"`);
        }
      }
      continue;
    }

    if (rec.type === 'query') {
      try {
        // Render through DuckDB's own ::VARCHAR, as native sqllogictest does,
        // rather than through Arrow-JS (which mis-renders DECIMAL: 1.0 -> "10"),
        // with each column's type from typeof() in the same statement: the
        // first half of every row is the values, the second half their types.
        // Only a query the wrap cannot bind or parse (PRAGMA, SHOW, ...) is
        // retried raw and compared by type letter; a runtime error is the
        // query's own and is reported, not run a second time.
        const inner = rec.sql.trim().replace(/;\s*$/, '');
        let rows;
        let sqlTypes = null;
        try {
          const wide = await run(`SELECT COLUMNS(*)::VARCHAR, typeof(COLUMNS(*)) FROM (\n${inner}\n) AS _wrap`);
          const width = wide.length ? wide[0].length / 2 : 0;
          rows = wide.map((r) => r.slice(0, width));
          if (wide.length) sqlTypes = wide[0].slice(width).map(String);
        } catch (err) {
          if (err instanceof RecordTimeout || !RETRY_RAW.test(String(err.message || err))) throw err;
          result.raw++;
          rows = await run(rec.sql);
        }
        const cmp = compareQuery(rec, rows, sqlTypes);
        if (cmp.ok) result.passed++;
        else fail(rec, cmp.reason);
      } catch (err) {
        if (err instanceof RecordTimeout) throw err;
        fail(rec, `query threw: ${err.message || err}`);
      }
      continue;
    }
  }

  if (log && result.failed > 0) {
    for (const f of result.failures) {
      log(`    ✗ ${f.where}: ${f.reason}\n      SQL: ${f.sql.replace(/\n/g, ' ').slice(0, 160)}`);
    }
  }
  return result;
}

// ---- Self-test -----------------------------------------------------------

export async function selfTest() {
  const q = (types, expected, sortMode = 'nosort') => ({ types, sortMode, expected });
  const cases = [
    // [name, record, rows, sqlTypes, expect ok]
    ['integers exact: 100000 vs 100001', q('I', ['100000']), [['100001']], ['BIGINT'], false],
    ['integers exact, no type info: 100000 vs 100001', q('I', ['100000']), [['100001']], null, false],
    ['BIGINT precision: 9007199254740993 vs ...992', q('I', ['9007199254740993']), [['9007199254740992']], ['BIGINT'], false],
    ['BIGINT equal beyond 2^53', q('I', ['9007199254740993']), [['9007199254740993']], ['BIGINT'], true],
    ['text exact: "001" vs "1"', q('T', ['001']), [['1']], ['VARCHAR'], false],
    ['text exact, no type info: "001" vs "1"', q('T', ['001']), [['1']], null, false],
    ['text exact under query I: "001" vs "1"', q('I', ['001']), [['1']], ['VARCHAR'], false],
    ['shape: rows [[1],[2]] vs "1\\t2" under query II', q('II', ['1\t2']), [['1'], ['2']], null, false],
    ['shape: one column returned, II declared', q('II', ['1', '2']), [['1'], ['2']], null, false],
    ['row count: 2 expected, 3 returned', q('I', ['1', '2']), [['1'], ['2'], ['3']], ['INTEGER'], false],
    ['row count: 3 expected, 2 returned', q('I', ['1', '2', '3']), [['1'], ['2']], ['INTEGER'], false],
    ['two columns, tab separated', q('II', ['1\t2']), [['1', '2']], ['INTEGER', 'INTEGER'], true],
    ['two columns, one value per line', q('II', ['1', '2']), [['1', '2']], ['INTEGER', 'INTEGER'], true],
    ['double within 1e-9', q('R', ['0.30000000000000004']), [['0.3']], ['DOUBLE'], true],
    ['double off by 1e-6 relative', q('R', ['1.000001']), [['1.0']], ['DOUBLE'], false],
    ['double 100000 vs 100001', q('R', ['100000.0']), [['100001.0']], ['DOUBLE'], false],
    ['double under query I', q('I', ['0.5']), [['0.5000000000000001']], ['DOUBLE'], true],
    ['double nan vs nan', q('R', ['nan']), [['nan']], ['DOUBLE'], true],
    ['double inf vs -inf', q('R', ['inf']), [['-inf']], ['DOUBLE'], false],
    ['DECIMAL scale: 1.0 vs 1.00', q('R', ['1.0']), [['1.00']], ['DECIMAL(18,2)'], true],
    ['boolean: query I true vs 1', q('I', ['1']), [['true']], ['BOOLEAN'], true],
    ['boolean: true vs false', q('I', ['true']), [['false']], ['BOOLEAN'], false],
    ['NULL vs NULL', q('I', ['NULL']), [[null]], ['INTEGER'], true],
    ['NULL vs 0', q('I', ['NULL']), [['0']], ['INTEGER'], false],
    ['empty string renders as (empty)', q('T', ['(empty)']), [['']], ['VARCHAR'], true],
    ['empty string vs NULL', q('T', ['(empty)']), [[null]], ['VARCHAR'], false],
    ['list text exact', q('T', ['[1, 2]']), [['[1,2]']], ['BIGINT[]'], false],
    ['struct with doubles is text: no tolerance', q('T', ["{'a': 0.1}"]), [["{'a': 0.10000000000000002}"]], ['STRUCT(a DOUBLE)'], false],
    ['rowsort sorts both sides', q('II', ['b\t1', 'a\t2'], 'rowsort'), [['a', '2'], ['b', '1']], ['VARCHAR', 'INTEGER'], true],
    ['nosort keeps order', q('I', ['2', '1']), [['1'], ['2']], ['INTEGER'], false],
    ['valuesort', q('I', ['3', '1', '2'], 'valuesort'), [['2'], ['3'], ['1']], ['INTEGER'], true],
    ['hash match', q('I', ['2 values hashing to ' + createHash('md5').update('1\n2\n').digest('hex')]), [['1'], ['2']], ['INTEGER'], true],
    ['hash mismatch', q('I', ['2 values hashing to ' + createHash('md5').update('1\n3\n').digest('hex')]), [['1'], ['2']], ['INTEGER'], false],
  ];
  const failures = [];
  for (const [name, rec, rows, types, want] of cases) {
    const got = compareQuery(rec, rows, types).ok;
    if (got !== want) failures.push(`${name}: expected ${want ? 'accept' : 'reject'}, got ${got ? 'accept' : 'reject'}`);
  }

  let checks = 0;
  const check = (cond, msg) => { checks++; if (!cond) failures.push(msg); };

  // Parser: the `statement error` message is split from the SQL; unknown
  // keywords become failing records with their line.
  const parsed = parseTest('statement error\nSELECT 1/0;\n----\nboom\n\nskipif wasm\nquery I\nSELECT 1\n\nstatement maybe\nCREATE TRIGGER t\n----\nParser Error\n');
  check(parsed[0].sql === 'SELECT 1/0;' && parsed[0].expectedError === 'boom', 'statement error: SQL/message not split');
  check(parsed.filter((r) => r.type === 'unknown').map((r) => r.line).join(',') === '6', 'unknown keyword: expected line 6');
  const maybe = parsed.find((r) => r.kind === 'maybe');
  check(maybe && maybe.sql === 'CREATE TRIGGER t' && maybe.expectedError === 'Parser Error' && maybe.line === 10, 'statement maybe: not parsed');

  // foreach / loop expand, substitute in the SQL and the expected results,
  // nest, and keep the line numbers of the file.
  const looped = parseTest("foreach m a b\nloop i 0 2\nquery T\nSELECT '${m}${i}'\n----\n${m}${i}\n\nendloop\nendloop\n");
  check(looped.map((r) => r.sql).join(',') === "SELECT 'a0',SELECT 'a1',SELECT 'b0',SELECT 'b1'", `loop expansion: got ${looped.map((r) => r.sql).join(',')}`);
  check(looped.every((r) => r.line === 3) && looped[3].expected[0] === 'b1', 'loop expansion: line numbers / expected substitution');
  check(parseTest('query I\nSELECT 1\n\nforeach x a b\nquery I\nSELECT 1\n')[0].line === 4, 'unterminated foreach: not reported at its line');
  check(parseTest('foreach x <integral>\nquery I\nSELECT 1\nendloop\n')[0].type === 'unknown', 'foreach value group: not rejected');
  check(parseTest('endloop\n')[0].type === 'unknown', 'stray endloop: not rejected');

  // Error-message matching.
  check(errorMatches('<REGEX>:.*cap.*', 'x cap y') && !errorMatches('<REGEX>:cap', 'x cap y'), 'REGEX: whole-message match');
  check(errorMatches('<!REGEX>:.*boom.*', 'x cap y') && !errorMatches('<!REGEX>:.*cap.*', 'x cap y'), '!REGEX');

  // Statement outcomes and the timeout, against a fake engine.
  const engine = async (sql) => {
    if (sql.startsWith('FAIL')) throw new Error(sql.slice(5));
    if (sql.startsWith('HANG')) return new Promise(() => {});
    return [];
  };
  const outcome = (text) => runRecords(parseTest(text), engine, { file: 'self', timeoutMs: 50 });
  const st = async (text, passed, failed) => {
    const r = await outcome(text);
    check(r.passed === passed && r.failed === failed, `${JSON.stringify(text)}: expected ${passed}/${failed} passed/failed, got ${r.passed}/${r.failed}`);
  };
  await st('statement error\nFAIL Invalid Input Error: x must be > 0\n----\nx must be > 0\n', 1, 0);
  await st('statement error\nFAIL Invalid Input Error: x must be > 0\n----\ny must be > 0\n', 0, 1);
  await st('statement error\nSELECT 1\n----\nboom\n', 0, 1);
  await st('statement error\nFAIL INTERNAL Error: assertion failed\n', 0, 1);
  await st('statement maybe\nSELECT 1\n----\nParser Error\n', 1, 0);
  await st('statement maybe\nFAIL Parser Error: syntax error at TRIGGER\n----\nParser Error\n', 1, 0);
  await st('statement maybe\nFAIL Binder Error: no such table\n----\nParser Error\n', 0, 1);
  await st('statement ok\nFAIL Binder Error: no such table\n', 0, 1);
  await st('skipif wasm\nstatement ok\nSELECT 1\n', 0, 1);
  let timedOut = false;
  try { await outcome('statement ok\nHANG\n'); } catch (err) { timedOut = err instanceof RecordTimeout && /self:1/.test(err.message); }
  check(timedOut, 'a hung record must raise RecordTimeout naming file:line');

  if (failures.length) {
    for (const f of failures) process.stderr.write(`SELF-TEST FAILURE: ${f}\n`);
    return 1;
  }
  process.stdout.write(`sqllogic self-test: ${cases.length} comparator cases and ${checks} parser/runner checks passed\n`);
  return 0;
}

if (process.argv[1] === fileURLToPath(import.meta.url) && process.argv.includes('--self-test')) {
  process.exit(await selfTest());
}
