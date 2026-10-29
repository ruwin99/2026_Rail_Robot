"use client";

import { useEffect, useMemo, useState, type FormEvent } from "react";

type PlannerCategory = "port_operation" | "maintenance" | "inspection" | "safety";
type PlannerStatus = "scheduled" | "in_progress" | "done";

type PlannerEvent = {
  id: string;
  date: string;
  startTime: string;
  endTime: string;
  title: string;
  category: PlannerCategory;
  location: string;
  notes: string;
  status: PlannerStatus;
  source: "starter" | "user";
};

type PlannerForm = Omit<PlannerEvent, "id" | "status" | "source">;
export type InspectionDayState = "clean" | "problem";

const STORAGE_KEY = "rail-dashboard:port-planner:v1";
const WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"];
const CATEGORY_LABELS: Record<PlannerCategory, string> = {
  port_operation: "항구 운영",
  maintenance: "유지보수",
  inspection: "레일 점검",
  safety: "안전 계획",
};

const localDate = (date: Date) => {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
};

const offsetDate = (base: Date, offset: number) => {
  const next = new Date(base);
  next.setHours(12, 0, 0, 0);
  next.setDate(next.getDate() + offset);
  return localDate(next);
};

const starterEvents = (): PlannerEvent[] => {
  const today = new Date();
  return [
    { id: "starter-port", date: offsetDate(today, 0), startTime: "08:30", endTime: "10:00", title: "제3부두 하역 작업 사전점검", category: "port_operation", location: "제3부두 작업구역", notes: "작업 동선과 레일 주변 장애물을 확인합니다.", status: "scheduled", source: "starter" },
    { id: "starter-inspection", date: offsetDate(today, 1), startTime: "10:00", endTime: "11:30", title: "주행 레일 정기 표면점검", category: "inspection", location: "B0441 점검구간", notes: "초기 10개 Rail Section 검사 후 청소·재검사를 진행합니다.", status: "scheduled", source: "starter" },
    { id: "starter-maintenance", date: offsetDate(today, 3), startTime: "14:00", endTime: "16:00", title: "레일 체결부 유지보수", category: "maintenance", location: "북측 주행 레일", notes: "체결부 풀림과 이물질을 확인하고 정비 기록을 남깁니다.", status: "scheduled", source: "starter" },
    { id: "starter-safety", date: offsetDate(today, 5), startTime: "09:00", endTime: "10:00", title: "비상정지·통신 단선 안전시험", category: "safety", location: "축소 테스트베드", notes: "실제 설비가 아닌 승인된 시험 환경에서만 수행합니다.", status: "scheduled", source: "starter" },
    { id: "starter-coating", date: offsetDate(today, 8), startTime: "13:30", endTime: "15:00", title: "방청 도장 보수 구역 확인", category: "maintenance", location: "서측 레일 구간", notes: "도장 손상 후보를 확인하고 보수 필요 구간을 지정합니다.", status: "scheduled", source: "starter" },
  ];
};

const blankForm = (date: string): PlannerForm => ({
  date,
  startTime: "09:00",
  endTime: "10:00",
  title: "",
  category: "port_operation",
  location: "",
  notes: "",
});

const monthLabel = (month: string) => {
  const [year, value] = month.split("-").map(Number);
  return `${year}년 ${value}월`;
};

const shiftMonth = (month: string, offset: number) => {
  const [year, value] = month.split("-").map(Number);
  const shifted = new Date(year, value - 1 + offset, 1);
  return `${shifted.getFullYear()}-${String(shifted.getMonth() + 1).padStart(2, "0")}`;
};

const calendarCells = (month: string) => {
  const [year, value] = month.split("-").map(Number);
  const leading = (new Date(year, value - 1, 1).getDay() + 6) % 7;
  const count = new Date(year, value, 0).getDate();
  return [...Array<null>(leading).fill(null), ...Array.from({ length: count }, (_, index) => index + 1)];
};

export function PortPlanner({
  inspectionDayStates,
  selectedDate,
  onSelectDate,
}: {
  inspectionDayStates: Record<string, InspectionDayState>;
  selectedDate: string;
  onSelectDate: (date: string) => void;
}) {
  const today = localDate(new Date());
  const activeDate = selectedDate || today;
  const [events, setEvents] = useState<PlannerEvent[]>([]);
  const [storageReady, setStorageReady] = useState(false);
  const [monthOffset, setMonthOffset] = useState(0);
  const [form, setForm] = useState<PlannerForm>(() => blankForm(activeDate));
  const [editingId, setEditingId] = useState<string | null>(null);
  const month = shiftMonth(activeDate.slice(0, 7), monthOffset);

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      try {
        const saved = window.localStorage.getItem(STORAGE_KEY);
        const parsed: unknown = saved ? JSON.parse(saved) : null;
        setEvents(Array.isArray(parsed) ? parsed as PlannerEvent[] : starterEvents());
      } catch {
        setEvents(starterEvents());
      }
      setStorageReady(true);
    }, 0);
    return () => window.clearTimeout(timeout);
  }, []);

  useEffect(() => {
    if (storageReady) window.localStorage.setItem(STORAGE_KEY, JSON.stringify(events));
  }, [events, storageReady]);

  const eventCounts = useMemo(() => {
    const counts = new Map<string, number>();
    events.forEach((event) => counts.set(event.date, (counts.get(event.date) ?? 0) + 1));
    return counts;
  }, [events]);

  const scheduledInspectionDates = useMemo(
    () => new Set(events
      .filter((event) => event.category === "inspection" && event.status !== "done")
      .map((event) => event.date)),
    [events],
  );

  const selectedEvents = useMemo(
    () => events.filter((event) => event.date === activeDate).sort((a, b) => a.startTime.localeCompare(b.startTime)),
    [activeDate, events],
  );
  const activeInspectionState = inspectionDayStates[activeDate];

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!form.title.trim() || !form.date) return;
    if (editingId) {
      setEvents((current) => current.map((item) => item.id === editingId
        ? { ...item, ...form, title: form.title.trim(), location: form.location.trim(), notes: form.notes.trim(), source: "user" }
        : item));
    } else {
      setEvents((current) => [...current, {
        ...form,
        id: globalThis.crypto?.randomUUID?.() ?? `plan-${Date.now()}`,
        title: form.title.trim(),
        location: form.location.trim(),
        notes: form.notes.trim(),
        status: "scheduled",
        source: "user",
      }]);
    }
    onSelectDate(form.date);
    setMonthOffset(0);
    setEditingId(null);
    setForm(blankForm(form.date));
  };

  const edit = (item: PlannerEvent) => {
    setEditingId(item.id);
    onSelectDate(item.date);
    setMonthOffset(0);
    setForm({ date: item.date, startTime: item.startTime, endTime: item.endTime, title: item.title, category: item.category, location: item.location, notes: item.notes });
  };

  const cancelEdit = () => {
    setEditingId(null);
    setForm(blankForm(activeDate));
  };

  const selectDay = (day: number) => {
    const date = `${month}-${String(day).padStart(2, "0")}`;
    setMonthOffset(0);
    if (!editingId) setForm((current) => ({ ...current, date }));
    onSelectDate(date);
  };

  const selectToday = () => {
    setMonthOffset(0);
    if (!editingId) setForm((current) => ({ ...current, date: today }));
    onSelectDate(today);
  };

  return (
    <section className="planner-section" aria-labelledby="planner-title">
      <div className="planner-heading">
        <div>
          <p className="section-kicker">PORT OPERATIONS PLANNER</p>
          <h2 id="planner-title">항만 운영 플래너</h2>
          <p>항구 작업, 레일 점검, 유지보수와 안전 일정을 한 날짜 기준으로 관리합니다.</p>
        </div>
        <div className="planner-summary">
          <span>선택한 날짜</span>
          <strong>{activeDate.replaceAll("-", ".")}</strong>
          <small>일정 {selectedEvents.length}건 · 점검 결과 {activeInspectionState === "clean" ? "깨끗함" : activeInspectionState === "problem" ? "문제 있음" : "없음"}</small>
        </div>
      </div>

      <div className="planner-layout">
        <div className="planner-calendar">
          <div className="calendar-toolbar">
            <button type="button" aria-label="이전 달" onClick={() => setMonthOffset((current) => current - 1)}>←</button>
            <strong>{monthLabel(month)}</strong>
            <div>
              <button type="button" onClick={selectToday}>오늘</button>
              <button type="button" aria-label="다음 달" onClick={() => setMonthOffset((current) => current + 1)}>→</button>
            </div>
          </div>
          <div className="calendar-weekdays" aria-hidden="true">
            {WEEKDAYS.map((day) => <span key={day}>{day}</span>)}
          </div>
          <div className="calendar-grid">
            {calendarCells(month).map((day, index) => day == null ? <span className="calendar-blank" key={`blank-${index}`} /> : (() => {
              const date = `${month}-${String(day).padStart(2, "0")}`;
              const count = eventCounts.get(date) ?? 0;
              const inspectionState = inspectionDayStates[date];
              const hasScheduledInspection = scheduledInspectionDates.has(date);
              const calendarState = inspectionState === "problem" ? "result-problem"
                : inspectionState === "clean" ? "result-clean"
                  : hasScheduledInspection ? "scheduled-inspection" : "";
              const stateLabel = inspectionState === "problem" ? ", 점검 결과 문제 있음"
                : inspectionState === "clean" ? ", 점검 결과 깨끗함"
                  : hasScheduledInspection ? ", 정기점검 예정" : "";
              return (
                <button
                  type="button"
                  key={date}
                  className={`${date === activeDate ? "selected" : ""} ${date === today ? "today" : ""} ${calendarState}`}
                  onClick={() => selectDay(day)}
                  aria-label={`${date}, 일정 ${count}건${stateLabel}`}
                >
                  <span>{day}</span>
                  <small>{count ? `${count}건` : ""}</small>
                  {calendarState ? <i className={`calendar-state-mark ${calendarState}`} aria-hidden="true" /> : null}
                </button>
              );
            })())}
          </div>
          <div className="calendar-legend">
            <span><i className="clean-dot" /> 검사 결과 깨끗함</span>
            <span><i className="scheduled-dot" /> 정기점검 예정</span>
            <span><i className="problem-dot" /> 문제 발견</span>
          </div>
        </div>

        <div className="planner-agenda">
          <div className="agenda-heading"><span className="tiny-label">DAILY PLAN</span><strong>{activeDate.slice(5).replace("-", "/")} 일정</strong></div>
          <div className="agenda-list">
            {selectedEvents.length ? selectedEvents.map((item) => (
              <article className={`agenda-item ${item.category} ${item.status === "done" ? "done" : ""}`} key={item.id}>
                <div className="agenda-time"><strong>{item.startTime}</strong><span>{item.endTime}</span></div>
                <div className="agenda-copy">
                  <div><span className="category-chip">{CATEGORY_LABELS[item.category]}</span>{item.source === "starter" ? <span className="starter-chip">예시</span> : null}</div>
                  <h3>{item.title}</h3>
                  <p>{item.location || "장소 미지정"}{item.notes ? ` · ${item.notes}` : ""}</p>
                </div>
                <div className="agenda-actions">
                  <button type="button" onClick={() => setEvents((current) => current.map((event) => event.id === item.id ? { ...event, status: event.status === "done" ? "scheduled" : "done" } : event))}>{item.status === "done" ? "되돌리기" : "완료"}</button>
                  <button type="button" onClick={() => edit(item)}>수정</button>
                  <button type="button" className="danger" onClick={() => setEvents((current) => current.filter((event) => event.id !== item.id))}>삭제</button>
                </div>
              </article>
            )) : <div className="agenda-empty">이 날짜에는 등록된 일정이 없습니다.</div>}
          </div>
        </div>

        <form className="planner-form" onSubmit={submit}>
          <div className="form-heading"><span className="tiny-label">{editingId ? "EDIT PLAN" : "NEW PLAN"}</span><strong>{editingId ? "일정 수정" : "일정 추가"}</strong></div>
          <label>일정명<input required value={form.title} onChange={(event) => setForm({ ...form, title: event.target.value })} placeholder="예: 2부두 레일 정기점검" /></label>
          <div className="form-row">
            <label>날짜<input required type="date" value={form.date} onChange={(event) => setForm({ ...form, date: event.target.value })} /></label>
            <label>분류<select value={form.category} onChange={(event) => setForm({ ...form, category: event.target.value as PlannerCategory })}>{Object.entries(CATEGORY_LABELS).map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label>
          </div>
          <div className="form-row">
            <label>시작<input required type="time" value={form.startTime} onChange={(event) => setForm({ ...form, startTime: event.target.value })} /></label>
            <label>종료<input required type="time" value={form.endTime} onChange={(event) => setForm({ ...form, endTime: event.target.value })} /></label>
          </div>
          <label>장소<input value={form.location} onChange={(event) => setForm({ ...form, location: event.target.value })} placeholder="예: 제3부두 북측 레일" /></label>
          <label>메모<textarea value={form.notes} onChange={(event) => setForm({ ...form, notes: event.target.value })} placeholder="작업 범위, 담당자 확인사항 등을 입력하세요." /></label>
          <div className="form-actions"><button type="submit">{editingId ? "수정 저장" : "일정 등록"}</button>{editingId ? <button type="button" className="secondary" onClick={cancelEdit}>취소</button> : null}</div>
          <p className="storage-note">현재 일정은 이 브라우저에 저장됩니다. 온라인 DB 연결 전에는 다른 PC와 자동 공유되지 않습니다.</p>
        </form>
      </div>
    </section>
  );
}
