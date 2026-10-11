import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { Workbook } from '@oai/artifact-tool';

const stage = path.dirname(new URL(import.meta.url).pathname.replace(/^\/(?=[A-Za-z]:)/, ''));
const output = 'D:/SummerResearch/output/statistical_analysis_328_20261010/csv_replication';
const tables = JSON.parse(await fs.readFile(path.join(stage, 'tables.json'), 'utf8'));
await fs.mkdir(output, { recursive: true });

function field(value) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'number' && !Number.isFinite(value)) throw new Error('Nonfinite CSV value');
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

// CSV export uses the documented public range values, preserving numeric
// precision and blank cells. No worksheet display formatting enters the files.
async function exportTable(table) {
  const wb = Workbook.create();
  const sheet = wb.worksheets.add('Data');
  const matrix = [table.headers, ...table.rows];
  const range = sheet.getRangeByIndexes(0, 0, matrix.length, table.headers.length);
  // Force ISO provenance timestamps to literal text, retaining Chicago offset
  // and microseconds instead of allowing spreadsheet date conversion.
  const isTimestamp = v => typeof v === 'string' && /^\d{4}-\d{2}-\d{2}T/.test(v);
  range.values = matrix.map(row => row.map(v => isTimestamp(v) ? `'${v}` : v));
  wb.recalculate();
  const exported = range.values.map((row, i) => row.map((v, j) =>
    isTimestamp(matrix[i][j]) && v === `'${matrix[i][j]}` ? v.slice(1) : v));
  if (exported.length !== matrix.length) throw new Error(`Row loss: ${table.filename}`);
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < table.headers.length; j++) {
      if ((exported[i][j] ?? null) !== (matrix[i][j] ?? null)) {
        throw new Error(`Changed input at ${table.filename}, row ${i}, column ${j}: ${JSON.stringify(matrix[i][j])} -> ${JSON.stringify(exported[i][j])}`);
      }
    }
  }
  await wb.inspect({kind:'region', sheetId:sheet.name, range:'A1:D3', maxChars:1000});
  const csv = exported.map(row => row.map(field).join(',')).join('\r\n') + '\r\n';
  const target = path.join(output, table.filename);
  await fs.writeFile(target, csv, 'utf8');
  return [table.filename, table.rows.length, table.headers.length,
          crypto.createHash('sha256').update(csv, 'utf8').digest('hex')];
}

const manifest = [];
for (const table of tables) {
  manifest.push(await exportTable(table));
  console.log(`${table.filename}: ${table.rows.length} rows`);
}
await exportTable({filename:'manifest.csv',headers:['file','rows','columns','sha256'],rows:manifest});
console.log('Exported 21 CSV files; all cells preserve their original scalar values.');
