import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getFunctions } from "firebase/functions";
import { getStorage } from "firebase/storage";

// [공통 Firebase 인스턴스] Auth는 사용자 인증, Firestore는 기록/상태/시그널링, Storage는 이미지 파일을 맡는다.
// 아래는 웹 앱의 프로젝트 연결 설정이며 사용자 로그인이나 데이터 접근 권한을 부여하는 설정은 아니다.
// permission-denied를 추적할 때 프로젝트/로그인 UID와 배포된 rules를 함께 확인한다.
const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};

const firebaseApp = initializeApp(firebaseConfig);

// 같은 앱 인스턴스에서 세 SDK를 꺼내므로 소비자들은 이 모듈의 auth/db/storage를 공유한다.
// SDP/ICE 처리 순서는 webrtc-signaling.ts, 검사 JSON/이미지 처리 순서는 firebase-data.ts를 확인한다.
export const auth = getAuth(firebaseApp);
export const db = getFirestore(firebaseApp);
export const storage = getStorage(firebaseApp);
// TURN 임시 계정 발급 함수(functions/index.js)와 같은 서울 리전.
export const functions = getFunctions(firebaseApp, "asia-northeast3");
export const googleProvider = new GoogleAuthProvider();

googleProvider.setCustomParameters({ prompt: "select_account" });
