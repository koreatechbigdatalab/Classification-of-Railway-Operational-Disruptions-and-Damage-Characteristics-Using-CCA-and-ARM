import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookPath = path.resolve(
  "outputs",
  "kaia-k6-visualization-data",
  "KAIA_DA_k6_visualization_dataset.xlsx",
);

const fileBytes = await fs.readFile(workbookPath);
if (fileBytes.length < 50000) throw new Error(`Unexpected workbook size: ${fileBytes.length}`);
const sha256 = crypto.createHash("sha256").update(fileBytes).digest("hex");

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const sheetNames = workbook.worksheets.items.map((sheet) => sheet.name);
const expectedSheets = ["플롯 요약", "사례별 데이터", "지연 long", "피해액 long"];
if (JSON.stringify(sheetNames) !== JSON.stringify(expectedSheets)) {
  throw new Error(`Unexpected sheets: ${JSON.stringify(sheetNames)}`);
}

const summary = workbook.worksheets.getItem("플롯 요약");
const cases = workbook.worksheets.getItem("사례별 데이터");
const delayLong = workbook.worksheets.getItem("지연 long");
const damageLong = workbook.worksheets.getItem("피해액 long");

const summaryValues = summary.getRange("A7:AB12").values;
const summaryFormulas = summary.getRange("D7:E12").formulas;
if (summaryValues.length !== 6 || summaryValues[0].length !== 28) {
  throw new Error("Summary dimensions are not 6 x 28.");
}
if (summaryFormulas[0][0] !== "=B7-C7" || summaryFormulas[0][1] !== "=C7/B7") {
  throw new Error(`Summary formulas missing: ${JSON.stringify(summaryFormulas[0])}`);
}

const caseValues = cases.getRange("A7:G745").values;
const delayValues = delayLong.getRange("A7:D1484").values;
const damageValues = damageLong.getRange("A7:D745").values;
if (caseValues.length !== 739) throw new Error(`Case row count: ${caseValues.length}`);
if (delayValues.length !== 1478) throw new Error(`Delay-long row count: ${delayValues.length}`);
if (damageValues.length !== 739) throw new Error(`Damage-long row count: ${damageValues.length}`);

const clusterCounts = {};
for (const row of caseValues) {
  const cluster = Number(row[2]);
  clusterCounts[cluster] = (clusterCounts[cluster] ?? 0) + 1;
}
const expectedCounts = { 1: 337, 2: 132, 3: 97, 4: 90, 5: 48, 6: 35 };
if (JSON.stringify(clusterCounts) !== JSON.stringify(expectedCounts)) {
  throw new Error(`Cluster counts mismatch: ${JSON.stringify(clusterCounts)}`);
}

const expectedDamagePositive = [292, 32, 50, 44, 9, 18];
for (let index = 0; index < 6; index += 1) {
  if (Number(summaryValues[index][2]) !== expectedDamagePositive[index]) {
    throw new Error(`Damage-positive count mismatch at cluster ${index + 1}`);
  }
}

const formulaErrors = await workbook.inspect({
  kind: "match",
  sheetId: "플롯 요약",
  range: "A1:AB24",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "verification formula error scan",
  maxChars: 3000,
});
if (!formulaErrors.ndjson.includes("matched 0 entries")) {
  throw new Error(`Formula errors found: ${formulaErrors.ndjson}`);
}

const renderSpecs = [
  ["플롯 요약", "A1:AB24"],
  ["사례별 데이터", "A1:G20"],
  ["지연 long", "A1:D20"],
  ["피해액 long", "A1:D20"],
];
const renderSizes = {};
for (const [sheetName, range] of renderSpecs) {
  const blob = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  const bytes = new Uint8Array(await blob.arrayBuffer());
  if (bytes.length < 10000) throw new Error(`Render too small for ${sheetName}: ${bytes.length}`);
  renderSizes[sheetName] = bytes.length;
}

console.log(JSON.stringify({
  workbookPath,
  bytes: fileBytes.length,
  sha256,
  sheetNames,
  rowCounts: { cases: caseValues.length, delayLong: delayValues.length, damageLong: damageValues.length },
  clusterCounts,
  formulaErrorMatches: 0,
  renderSizes,
}, null, 2));
