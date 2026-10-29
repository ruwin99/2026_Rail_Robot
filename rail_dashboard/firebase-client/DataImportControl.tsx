import { useRef, useState } from "react";
import { importLegacyInspectionFiles } from "./firebase-data";

export function DataImportControl({ onImported }: { onImported: () => Promise<void> }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [state, setState] = useState<"idle" | "uploading" | "done" | "error">("idle");
  const [message, setMessage] = useState("기존 테스트 기록 업로드");

  const importFiles = async (files: File[]) => {
    setState("uploading");
    try {
      const result = await importLegacyInspectionFiles(files, setMessage);
      setMessage(`점검 ${result.runCount}건 · 마스크 ${result.previewCount}개 적용`);
      setState("done");
      await onImported();
    } catch (error: unknown) {
      setMessage(error instanceof Error ? error.message : "데이터 업로드에 실패했습니다.");
      setState("error");
    } finally {
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  return (
    <div className={`sidebar-import ${state}`}>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept="application/json,image/png,image/jpeg"
        aria-label="기존 테스트 데이터 폴더 선택"
        onChange={(event) => void importFiles(Array.from(event.target.files ?? []))}
        {...({ webkitdirectory: "" } as Record<string, string>)}
      />
      <button type="button" disabled={state === "uploading"} onClick={() => inputRef.current?.click()}>
        {state === "uploading" ? "전송 중" : "테스트 DB 적용"}
      </button>
      <span>{message}</span>
    </div>
  );
}
