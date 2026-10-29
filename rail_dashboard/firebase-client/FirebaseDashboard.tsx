import { useCallback, useEffect, useRef, useState } from "react";
import type { InspectionExport } from "../app/data-contract";
import { InspectionDashboard } from "../app/InspectionDashboard";
import { AdminAuthControl } from "./AdminAuthControl";
import { DataImportControl } from "./DataImportControl";
import { loadServerInspectionData } from "./firebase-data";
import { LiveView } from "./LiveView";

export function FirebaseDashboard() {
  const [isAdmin, setIsAdmin] = useState(false);
  const [serverDocument, setServerDocument] = useState<InspectionExport | null>(null);
  const [serverState, setServerState] = useState<"loading" | "live" | "demo" | "error">("loading");
  const [serverMessage, setServerMessage] = useState<string | null>(null);
  const objectUrlsRef = useRef<string[]>([]);

  const replaceObjectUrls = useCallback((nextUrls: string[]) => {
    objectUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    objectUrlsRef.current = nextUrls;
  }, []);

  const refreshServerData = useCallback(async () => {
    try {
      const result = await loadServerInspectionData();
      replaceObjectUrls(result.objectUrls);
      setServerDocument(result.document);
      setServerState(result.document ? "live" : "demo");
      setServerMessage(result.document ? null : "서버에 저장된 점검 기록이 없어 예시 데이터를 표시합니다.");
    } catch (error: unknown) {
      replaceObjectUrls([]);
      setServerDocument(null);
      setServerState("error");
      setServerMessage(`서버 데이터를 읽지 못했습니다: ${error instanceof Error ? error.message : String(error)}`);
    }
  }, [replaceObjectUrls]);

  const handleAdminChange = useCallback((next: boolean) => {
    setIsAdmin(next);
    setServerDocument(null);
    setServerState("loading");
    setServerMessage(next ? "서울 Firestore에서 점검 기록을 불러오는 중입니다." : null);
    if (!next) replaceObjectUrls([]);
    else void refreshServerData();
  }, [refreshServerData, replaceObjectUrls]);

  const reloadServerData = () => {
    setServerState("loading");
    setServerMessage("서울 Firestore에서 점검 기록을 불러오는 중입니다.");
    return refreshServerData();
  };

  useEffect(() => () => replaceObjectUrls([]), [replaceObjectUrls]);

  return (
    <InspectionDashboard
      remoteDocument={isAdmin ? serverDocument : undefined}
      remoteState={isAdmin ? serverState : undefined}
      remoteMessage={isAdmin ? serverMessage : undefined}
      liveView={<LiveView isAdmin={isAdmin} />}
      adminAuthControl={(
        <AdminAuthControl onAdminChange={handleAdminChange}>
          {isAdmin ? <DataImportControl onImported={reloadServerData} /> : null}
        </AdminAuthControl>
      )}
    />
  );
}
