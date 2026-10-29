import { useEffect, useState, type ReactNode } from "react";
import { FirebaseError } from "firebase/app";
import { getRedirectResult, signInWithRedirect, signOut, type User } from "firebase/auth";
import { doc, getDoc } from "firebase/firestore";
import { auth, db, googleProvider } from "./firebase";

type AccessState = "loading" | "anonymous" | "checking" | "admin" | "denied" | "error";

const authErrorMessage = (error: unknown) => {
  if (!(error instanceof FirebaseError)) return "로그인 처리 중 오류가 발생했습니다.";
  return `Google 로그인을 완료하지 못했습니다. (${error.code})`;
};

export function AdminAuthControl({
  children,
  onAdminChange,
}: {
  children?: ReactNode;
  onAdminChange: (isAdmin: boolean) => void;
}) {
  const [accessState, setAccessState] = useState<AccessState>("loading");
  const [user, setUser] = useState<User | null>(null);
  const [message, setMessage] = useState("로그인 상태 확인 중");

  useEffect(() => {
    void getRedirectResult(auth).catch((error: unknown) => {
      setAccessState("anonymous");
      setMessage(authErrorMessage(error));
    });

    return auth.onAuthStateChanged((nextUser) => {
      setUser(nextUser);
      if (!nextUser) {
        onAdminChange(false);
        setAccessState("anonymous");
        setMessage("Google 계정으로 로그인");
        return;
      }

      setAccessState("checking");
      onAdminChange(false);
      setMessage("관리자 권한 확인 중");
      void getDoc(doc(db, "admin_access", "status"))
        .then(() => {
          onAdminChange(true);
          setAccessState("admin");
          setMessage(nextUser.displayName || "승인된 관리자");
        })
        .catch((error: unknown) => {
          if (error instanceof FirebaseError && error.code === "permission-denied") {
            onAdminChange(false);
            setAccessState("denied");
            setMessage("승인되지 않은 계정");
            return;
          }
          setAccessState("error");
          setMessage(`권한 확인 실패 (${error instanceof FirebaseError ? error.code : "unknown"})`);
        });
    });
  }, [onAdminChange]);

  const login = async () => {
    setAccessState("checking");
    setMessage("Google 로그인 여는 중");
    try {
      await signInWithRedirect(auth, googleProvider);
    } catch (error: unknown) {
      setAccessState("anonymous");
      setMessage(authErrorMessage(error));
    }
  };

  const logout = async () => {
    setAccessState("loading");
    setMessage("로그아웃 중");
    await signOut(auth);
  };

  return (
    <section className={`sidebar-auth ${accessState}`} aria-live="polite" aria-label="관리자 계정">
      <div className="sidebar-auth-row">
        <div className="sidebar-auth-copy">
          <small>ADMIN ACCESS</small>
          <strong>{accessState === "admin" ? "관리자 접속" : accessState === "denied" ? "권한 없음" : "관리자 로그인"}</strong>
          <span>{message}</span>
        </div>
        {accessState === "anonymous" ? (
          <button type="button" onClick={login}>로그인</button>
        ) : accessState === "admin" || accessState === "denied" ? (
          <button type="button" className="secondary" onClick={logout}>{user ? "로그아웃" : "닫기"}</button>
        ) : (
          <span className="auth-spinner" aria-hidden="true" />
        )}
      </div>
      {accessState === "admin" ? children : null}
    </section>
  );
}
