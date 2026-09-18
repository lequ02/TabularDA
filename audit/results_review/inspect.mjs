import fs from 'node:fs/promises';
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';
const wb = await SpreadsheetFile.importXlsx(await FileBlob.load('D:/SummerResearch/final_results.xlsx'));
console.log((await wb.inspect({kind:'workbook,sheet,table', maxChars:9000, tableMaxRows:5,tableMaxCols:12})).ndjson);
const sheets=[];
for(let i=0;i<20;i++) {
  let sheet;
  try { sheet=wb.worksheets.getItemAt(i); } catch {break;}
  if(!sheet) break;
  const range=sheet.getUsedRange();
  sheets.push({name:sheet.name,values:range.values,formulas:range.formulas});
}
await fs.writeFile('D:/SummerResearch/audit/results_review/workbook.json',JSON.stringify(sheets,null,2));
for(const sheet of sheets) {
  console.log(JSON.stringify({sheet:sheet.name,rows:sheet.values.length,columns:sheet.values[0]?.length}));
  sheet.values.forEach((row,i)=>{if(row.some(v=>v!==null && v!==''))console.log(JSON.stringify({row:i+1,values:row}));});
}
