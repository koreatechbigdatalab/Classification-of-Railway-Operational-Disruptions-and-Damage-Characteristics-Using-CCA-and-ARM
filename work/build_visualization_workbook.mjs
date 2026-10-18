import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const workspaceDir = path.resolve(".");
const inputPath = path.join(
  workspaceDir,
  "work",
  "plot_data_rebuild",
  "visualization_dataset.json",
);
const outputDir = path.join(workspaceDir, "outputs", "kaia-k6-visualization-data");
const previewDir = path.join(workspaceDir, "work", "plot_data_rebuild", "previews");
const outputPath = path.join(outputDir, "KAIA_DA_k6_visualization_dataset.xlsx");

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const payload = JSON.parse(await fs.readFile(inputPath, "utf8"));
const workbook = Workbook.create();
const fontName = "Arial";

const colors = {
  text: "#1F2937",
  muted: "#5B6470",
  slate: "#334155",
  slateLight: "#E8EDF3",
  orange: "#B95F24",
  orangeLight: "#F7E5D8",
  blue: "#355C7D",
  blueLight: "#DFEAF2",
  border: "#CBD5E1",
  white: "#FFFFFF",
};

function columnLetter(columnNumber) {
  let value = columnNumber;
  let output = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    output = String.fromCharCode(65 + remainder) + output;
    value = Math.floor((value - 1) / 26);
  }
  return output;
}

function recordsToMatrix(records, headers) {
  return records.map((record) => headers.map((header) => record[header] ?? null));
}

function applyBaseStyle(sheet, rangeAddress) {
  const range = sheet.getRange(rangeAddress);
  range.format.font = { name: fontName, size: 10, color: colors.text };
  range.format.verticalAlignment = "center";
}

function styleTitle(sheet, title, rangeAddress) {
  sheet.getRange(rangeAddress).values = [[title]];
  sheet.getRange(rangeAddress).format.font = {
    name: fontName,
    size: 14,
    bold: true,
    color: colors.text,
  };
}

function styleHeader(range, fill) {
  range.format.fill = fill;
  range.format.font = { name: fontName, size: 10, bold: true, color: colors.white };
  range.format.horizontalAlignment = "center";
  range.format.verticalAlignment = "center";
  range.format.wrapText = true;
  range.format.borders = {
    bottom: { style: "thin", color: colors.white },
  };
}

const summarySheet = workbook.worksheets.add("플롯 요약");
summarySheet.showGridLines = false;
summarySheet.tabColor = colors.orange;

const summaryHeaders = Object.keys(payload.summary[0]);
const summaryEndColumn = columnLetter(summaryHeaders.length);
styleTitle(summarySheet, "K=6 군집별 피해 및 운행지연 시각화 데이터", "A2");
summarySheet.getRange(`A3:${summaryEndColumn}3`).format.borders = {
  bottom: { style: "thin", color: colors.border },
};
summarySheet.getRange("A4").values = [[
  "각 행은 하나의 군집이다. 피해액 발생률, 양의 피해액 상자그림, 지연열차수와 최대지연시간 구간도에 필요한 값을 포함한다.",
]];
summarySheet.getRange("A4").format.font = {
  name: fontName,
  size: 9,
  italic: true,
  color: colors.muted,
};

summarySheet.getRange(`A6:${summaryEndColumn}6`).values = [summaryHeaders];
const summaryRows = recordsToMatrix(payload.summary, summaryHeaders);
for (const row of summaryRows) {
  row[3] = null;
  row[4] = null;
}
summarySheet.getRange(`A7:${summaryEndColumn}12`).values = summaryRows;
summarySheet.getRange("D7").formulas = [["=B7-C7"]];
summarySheet.getRange("D7:D12").fillDown();
summarySheet.getRange("E7").formulas = [["=C7/B7"]];
summarySheet.getRange("E7:E12").fillDown();

applyBaseStyle(summarySheet, `A2:${summaryEndColumn}25`);
styleTitle(summarySheet, "K=6 군집별 피해 및 운행지연 시각화 데이터", "A2");
styleHeader(summarySheet.getRange("A6:B6"), colors.slate);
styleHeader(summarySheet.getRange("C6:E6"), colors.orange);
styleHeader(summarySheet.getRange("F6:L6"), colors.blue);
styleHeader(summarySheet.getRange("M6:T6"), colors.slate);
styleHeader(summarySheet.getRange("U6:AB6"), colors.orange);
summarySheet.getRange("A6:AB6").format.rowHeight = 52;
summarySheet.getRange("A7:D12").setNumberFormat("#,##0");
summarySheet.getRange("E7:E12").setNumberFormat("0.0%");
summarySheet.getRange("F7:F12").setNumberFormat("#,##0");
summarySheet.getRange("G7:L12").setNumberFormat("#,##0.0000");
summarySheet.getRange("M7:M12").setNumberFormat("#,##0");
summarySheet.getRange("N7:T12").setNumberFormat("#,##0.0#");
summarySheet.getRange("U7:U12").setNumberFormat("#,##0");
summarySheet.getRange("V7:AB12").setNumberFormat("#,##0.0#");
summarySheet.getRange("A7:AB12").format.borders = {
  bottom: { style: "thin", color: colors.border },
};
summarySheet.getRange("A7:AB12").format.rowHeight = 22;

summarySheet.getRange("A15:B15").values = [["재현 정보", "정의 및 처리 기준"]];
styleHeader(summarySheet.getRange("A15:B15"), colors.slate);
summarySheet.getRange(`A16:B${15 + payload.metadata.length - 1}`).values = payload.metadata.slice(1);
summarySheet.getRange(`A16:B${15 + payload.metadata.length - 1}`).format.borders = {
  bottom: { style: "thin", color: colors.border },
};
summarySheet.getRange(`A16:A${15 + payload.metadata.length - 1}`).format.fill = colors.slateLight;
summarySheet.getRange(`A16:A${15 + payload.metadata.length - 1}`).format.font = {
  name: fontName,
  size: 10,
  bold: true,
  color: colors.text,
};
summarySheet.getRange(`B16:B${15 + payload.metadata.length - 1}`).format.wrapText = true;

summarySheet.getRange("A:A").format.columnWidth = 9;
summarySheet.getRange("B:D").format.columnWidth = 12;
summarySheet.getRange("E:E").format.columnWidth = 14;
summarySheet.getRange("F:F").format.columnWidth = 12;
summarySheet.getRange("G:L").format.columnWidth = 17;
summarySheet.getRange("M:T").format.columnWidth = 14;
summarySheet.getRange("U:AB").format.columnWidth = 16;
summarySheet.getRange("A15:A24").format.columnWidth = 22;
summarySheet.getRange("B15:B24").format.columnWidth = 62;

function addDataSheet({ name, title, sourceText, noteText, records, headerRow, tableName, numberFormats }) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  const headers = Object.keys(records[0]);
  const endColumn = columnLetter(headers.length);
  const dataStartRow = headerRow + 1;
  const dataEndRow = headerRow + records.length;

  sheet.getRange("A1").values = [[sourceText]];
  sheet.getRange("A1").format.font = {
    name: fontName,
    size: 9,
    italic: true,
    color: colors.muted,
  };
  styleTitle(sheet, title, "A3");
  sheet.getRange("A4").values = [[noteText]];
  sheet.mergeCells(`A4:${endColumn}4`);
  sheet.getRange("A4").format.font = {
    name: fontName,
    size: 9,
    italic: true,
    color: colors.muted,
  };
  sheet.getRange("A4").format.wrapText = true;
  sheet.getRange(`A4:${endColumn}4`).format.rowHeight = 28;

  sheet.getRange(`A${headerRow}:${endColumn}${headerRow}`).values = [headers];
  sheet.getRange(`A${dataStartRow}:${endColumn}${dataEndRow}`).values = recordsToMatrix(records, headers);
  applyBaseStyle(sheet, `A1:${endColumn}${dataEndRow}`);
  styleTitle(sheet, title, "A3");
  styleHeader(sheet.getRange(`A${headerRow}:${endColumn}${headerRow}`), colors.slate);
  sheet.getRange(`A${headerRow}:${endColumn}${headerRow}`).format.rowHeight = 34;
  sheet.getRange(`A${dataStartRow}:${endColumn}${dataEndRow}`).format.rowHeight = 19;
  sheet.getRange(`A${dataStartRow}:${endColumn}${dataEndRow}`).format.borders = {
    bottom: { style: "thin", color: "#E5E7EB" },
  };

  for (const [rangeAddress, formatCode] of Object.entries(numberFormats)) {
    sheet
      .getRange(rangeAddress.replaceAll("{start}", String(dataStartRow)).replaceAll("{end}", String(dataEndRow)))
      .setNumberFormat(formatCode);
  }
  sheet.tables.add(`A${headerRow}:${endColumn}${dataEndRow}`, true, tableName);
  sheet.freezePanes.freezeRows(headerRow);
  return { sheet, endColumn, dataEndRow };
}

const sourceText = "Source: 사용자 제공 철도사고·운행장애 원자료 및 KAIA_DA_k=6 분석 코드";

const caseData = addDataSheet({
  name: "사례별 데이터",
  title: "사례별 피해 및 지연 데이터",
  sourceText,
  noteText: "한 행은 분석에 포함된 운행장애 사례 1건이다. 피해액발생여부는 총 피해액이 0보다 크면 1이다.",
  records: payload.case_wide,
  headerRow: 6,
  tableName: "CaseDataTable",
  numberFormats: {
    "B{start}:C{end}": "#,##0",
    "D{start}:D{end}": "#,##0.0000",
    "E{start}:E{end}": "0",
    "F{start}:G{end}": "#,##0.0#",
  },
});
caseData.sheet.tabColor = colors.blue;
caseData.sheet.getRange("A:A").format.columnWidth = 14;
caseData.sheet.getRange("B:C").format.columnWidth = 12;
caseData.sheet.getRange("D:D").format.columnWidth = 20;
caseData.sheet.getRange("E:E").format.columnWidth = 16;
caseData.sheet.getRange("F:G").format.columnWidth = 22;

const delayLong = addDataSheet({
  name: "지연 long",
  title: "지연 지표 long 형식",
  sourceText,
  noteText: "노트북의 figure4_delay_comparison_data.csv와 같은 구조다. value 결측은 제거하지 않았다.",
  records: payload.delay_long,
  headerRow: 6,
  tableName: "DelayLongTable",
  numberFormats: {
    "B{start}:B{end}": "#,##0",
    "D{start}:D{end}": "#,##0.0#",
  },
});
delayLong.sheet.tabColor = colors.slate;
delayLong.sheet.getRange("A:A").format.columnWidth = 14;
delayLong.sheet.getRange("B:B").format.columnWidth = 10;
delayLong.sheet.getRange("C:C").format.columnWidth = 26;
delayLong.sheet.getRange("D:D").format.columnWidth = 14;

const damageLong = addDataSheet({
  name: "피해액 long",
  title: "피해액 지표 long 형식",
  sourceText,
  noteText: "노트북의 figure5_damage_comparison_data.csv와 같은 구조다. 0원 사례를 포함한 739건이다.",
  records: payload.damage_long,
  headerRow: 6,
  tableName: "DamageLongTable",
  numberFormats: {
    "B{start}:B{end}": "#,##0",
    "D{start}:D{end}": "#,##0.0000",
  },
});
damageLong.sheet.tabColor = colors.orange;
damageLong.sheet.getRange("A:A").format.columnWidth = 14;
damageLong.sheet.getRange("B:B").format.columnWidth = 10;
damageLong.sheet.getRange("C:C").format.columnWidth = 24;
damageLong.sheet.getRange("D:D").format.columnWidth = 16;

workbook.recalculate();

const summaryCheck = await workbook.inspect({
  kind: "table",
  range: "플롯 요약!A6:AB12",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 28,
  maxChars: 14000,
});
console.log("SUMMARY_CHECK");
console.log(summaryCheck.ndjson);

const errorCheck = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
  maxChars: 4000,
});
console.log("ERROR_CHECK");
console.log(errorCheck.ndjson);

const previewSpecs = [
  ["플롯 요약", "A1:AB24", "summary.png"],
  ["사례별 데이터", "A1:G20", "case-data.png"],
  ["지연 long", "A1:D20", "delay-long.png"],
  ["피해액 long", "A1:D20", "damage-long.png"],
];
for (const [sheetName, range, filename] of previewSpecs) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(path.join(previewDir, filename), new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const imported = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const savedSheets = await imported.inspect({ kind: "sheet", include: "id,name", maxChars: 4000 });
const savedSummary = await imported.inspect({
  kind: "table",
  range: "플롯 요약!A6:F12",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 6,
  maxChars: 6000,
});
console.log("SAVED_SHEETS");
console.log(savedSheets.ndjson);
console.log("SAVED_SUMMARY");
console.log(savedSummary.ndjson);
console.log(`OUTPUT_PATH=${outputPath}`);
