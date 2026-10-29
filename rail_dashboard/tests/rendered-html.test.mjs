import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the inspection dashboard", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>레일 표면 점검 대시보드<\/title>/);
  assert.match(html, /B0441/);
  assert.match(html, /RAIL CARE/);
  assert.match(html, /항만 레일을/);
  assert.match(html, /더 스마트하게 관리합니다/);
  assert.match(html, /dolphin-mark\.png/);
  assert.match(html, /종합 현황/);
  assert.match(html, /검출 기록/);
  assert.match(html, /항만 플래너/);
  assert.match(html, /검출 기록 열기/);
  assert.match(html, /선택한 날짜에는 점검 기록이 없습니다/);
  assert.doesNotMatch(html, /Your site is taking shape|Building your site/);
});

test("keeps the port planner usable without an online database", async () => {
  const source = await readFile(new URL("../app/PortPlanner.tsx", import.meta.url), "utf8");
  assert.match(source, /rail-dashboard:port-planner:v1/);
  assert.match(source, /제3부두 하역 작업 사전점검/);
  assert.match(source, /레일 체결부 유지보수/);
  assert.match(source, /비상정지·통신 단선 안전시험/);
  assert.match(source, /window\.localStorage\.setItem/);
  assert.match(source, /일정 등록/);
  assert.match(source, /수정 저장/);
  assert.match(source, /완료/);
  assert.match(source, /삭제/);
  assert.match(source, /검사 결과 깨끗함/);
  assert.match(source, /정기점검 예정/);
  assert.match(source, /문제 발견/);
  assert.match(source, /result-clean/);
  assert.match(source, /scheduled-inspection/);
  assert.match(source, /result-problem/);
});

test("keeps failure, manual-phase, and analysis-issue labels in the client UI", async () => {
  const [dashboardSource, contractSource] = await Promise.all([
    readFile(new URL("../app/InspectionDashboard.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/data-contract.ts", import.meta.url), "utf8"),
  ]);
  const source = `${dashboardSource}\n${contractSource}`;
  assert.match(source, /phase === "rescan" \? "청소 후" : "수동"/);
  assert.match(source, /실패 사유/);
  assert.match(source, /모델 분석 중 오류가 발생했습니다/);
  assert.match(source, /미측정 여부는 전·후 비교를 확인하세요/);
  assert.match(source, /\/data\/inspection-export\.json/);
  assert.match(source, /전체 레일 구간 위치/);
  assert.match(source, /\$\{segment\}번 레일 구간/);
  assert.doesNotMatch(source, /번 논리 구역/);
  assert.match(source, /type DashboardView = "overview" \| "inspections" \| "live" \| "planner"/);
  assert.match(source, /대시보드 기능 메뉴/);
  assert.match(source, /setActiveView\("inspections"\)/);
  assert.match(source, /setActiveView\("planner"\)/);
  assert.match(source, /검사할 크레인을 선택하세요/);
  assert.match(source, /aria-label="크레인 선택"/);
  assert.match(source, /크레인 미지정/);
  assert.match(source, /웹 공개 이미지를 사용한 데모 데이터/);
  assert.doesNotMatch(source, /시연용 더미 결과/);
  assert.match(source, /summaryIsDemo/);
  assert.match(source, /Piqsels · Public Domain/);
  assert.match(source, /Cracked Wall by Sherrie Thai/);
  assert.match(source, /role === "top" \? "윗면" : "옆면"/);
  assert.match(source, /cameraRole = capture\.camera_role/);
  assert.match(source, /aria-label="청소 전후 사진 비교"/);
  assert.match(source, /aria-label="표시할 청소 단계"/);
  assert.match(source, /detection\.phase === inspectionPhase/);
  assert.match(source, /issue\.phase === inspectionPhase/);
  assert.match(source, /청소 후 녹 오염률/);
  assert.match(source, /percentage > 0 && percentage < 0\.01 \? "<0\.01%"/);
});

test("refreshes synchronized inspection data while the dashboard remains open", async () => {
  const source = await readFile(
    new URL("../app/InspectionDashboard.tsx", import.meta.url),
    "utf8",
  );
  assert.match(source, /DATA_REFRESH_INTERVAL_MS = 5_000/);
  assert.match(source, /setInterval\(loadInspectionData, DATA_REFRESH_INTERVAL_MS\)/);
  assert.match(source, /cache: "no-store"/);
});

test("shows the latest Jetson GUI snapshot to signed-in dashboard users", async () => {
  const [dashboardSource, clientSource, storageSource] = await Promise.all([
    readFile(new URL("../app/InspectionDashboard.tsx", import.meta.url), "utf8"),
    readFile(new URL("../firebase-client/FirebaseDashboard.tsx", import.meta.url), "utf8"),
    readFile(new URL("../firebase-client/firebase-data.ts", import.meta.url), "utf8"),
  ]);
  assert.match(dashboardSource, /로봇 GUI 카메라 화면/);
  assert.match(dashboardSource, /label: "실시간 화면"/);
  assert.match(dashboardSource, /activeView === "live"/);
  assert.match(clientSource, /liveView=\{<LiveView isAdmin=\{isAdmin\} \/>\}/);
  assert.match(storageSource, /live\/jetson\/gui\.jpg/);
});

test("uses each run target for Rail Section progress and keeps legacy targets compatible", async () => {
  const [dashboardSource, contractSource, demoSource, styles, readme, syncSource] = await Promise.all([
    readFile(new URL("../app/InspectionDashboard.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/data-contract.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/demo-export.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../README.md", import.meta.url), "utf8"),
    readFile(new URL("../scripts/sync-dashboard-data.mjs", import.meta.url), "utf8"),
  ]);

  assert.match(contractSource, /railSectionTarget: run\.capture_target/);
  assert.match(contractSource, /initialRailSectionCount: summary\?\.initial_capture_count \?\? 0/);
  assert.match(contractSource, /rescanRailSectionCount: summary\?\.rescan_capture_count \?\? 0/);
  assert.match(dashboardSource, /Array\.from\(\{ length: railSectionTarget \}/);
  assert.match(dashboardSource, /repeat\(\$\{railSectionTarget\}, minmax\(90px, 1fr\)\)/);
  assert.match(dashboardSource, /initialRailSectionCount\}\/\{selectedRun\.railSectionTarget\}/);
  assert.match(dashboardSource, /rescanRailSectionCount\}\/\{selectedRun\.railSectionTarget\}/);
  assert.match(dashboardSource, /Rail Section 완료 후 계산/);
  assert.match(demoSource, /capture_target: 10/);
  assert.doesNotMatch(demoSource, /capture_target: 4/);
  assert.doesNotMatch(dashboardSource, /Array\.from\(\{ length: 4 \}|repeat\(4|4\+4/);
  assert.doesNotMatch(styles, /grid-template-columns:\s*repeat\(4/);
  assert.doesNotMatch(readme, /1~4|4\+4/);
  assert.match(readme, /목표가 4였던 기존 기록도 그대로/);
  assert.match(syncSource, /\$\{result\.captures\} camera records/);
  assert.doesNotMatch(syncSource, /\$\{result\.captures\} captures/);
});

test("uses real web photos for the clearly labelled demo records", async () => {
  const [demoSource, credits] = await Promise.all([
    readFile(new URL("../app/demo-export.ts", import.meta.url), "utf8"),
    readFile(new URL("../public/demo/IMAGE_SOURCES.md", import.meta.url), "utf8"),
  ]);
  assert.match(demoSource, /rust-web-demo\.jpg/);
  assert.match(demoSource, /crack-web-demo\.jpg/);
  assert.doesNotMatch(demoSource, /rust-mask-[ab]\.svg|crack-mask-[ab]\.svg/);
  assert.match(credits, /Public Domain/);
  assert.match(credits, /CC BY 2\.0/);
});

test("keeps the DB-ready schema and migration aligned", async () => {
  const [schema, inspectionMigration, plannerMigration, craneMigration, cameraRoleMigration, packageJson] = await Promise.all([
    readFile(new URL("../db/schema.ts", import.meta.url), "utf8"),
    readFile(new URL("../drizzle/0000_motionless_archangel.sql", import.meta.url), "utf8"),
    readFile(new URL("../drizzle/0001_port_planner_events.sql", import.meta.url), "utf8"),
    readFile(new URL("../drizzle/0002_inspection_run_crane.sql", import.meta.url), "utf8"),
    readFile(new URL("../drizzle/0003_capture_camera_role.sql", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);
  for (const table of [
    "inspection_runs",
    "model_provenance",
    "captures",
    "analyses",
    "artifacts",
    "run_summaries",
  ]) {
    assert.match(schema, new RegExp(table));
    assert.match(inspectionMigration, new RegExp(`CREATE TABLE .${table}.`));
  }
  assert.match(schema, /planner_events/);
  assert.match(plannerMigration, /CREATE TABLE `planner_events`/);
  assert.match(plannerMigration, /planner_events_date_idx/);
  assert.match(schema, /craneId: text\("crane_id"\)/);
  assert.match(schema, /inspection_runs_crane_date_idx/);
  assert.match(craneMigration, /ADD `crane_id` text/);
  assert.match(craneMigration, /inspection_runs_crane_date_idx/);
  assert.match(schema, /cameraRole: text\("camera_role", \{ enum: \["side", "top"\] \}\)\.notNull\(\)\.default\("side"\)/);
  assert.match(schema, /captures_run_phase_sequence_camera_role_uq/);
  assert.match(cameraRoleMigration, /DROP INDEX `captures_run_phase_sequence_uq`/);
  assert.match(cameraRoleMigration, /ADD `camera_role` text DEFAULT 'side' NOT NULL/);
  assert.match(cameraRoleMigration, /captures_run_phase_sequence_camera_role_uq/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});
