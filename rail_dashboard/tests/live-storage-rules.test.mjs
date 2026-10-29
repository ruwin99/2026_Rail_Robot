import { before, after, test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { initializeTestEnvironment, assertFails, assertSucceeds } from '@firebase/rules-unit-testing';
import { doc, setDoc } from 'firebase/firestore';
import { ref, uploadBytes, getBytes, deleteObject } from 'firebase/storage';

let env;
const cap = 'cap_' + 'a'.repeat(32);
const object = `robots/robot/captures/${cap}/raw.jpg`;
const admin = { email: 'admin1@example.invalid', email_verified: true };
const image = new Uint8Array([255, 216, 255, 217]);
const storage = (uid, claims = {}) => env.authenticatedContext(uid, claims).storage();
before(async () => {
  env = await initializeTestEnvironment({ projectId: 'demo-rail-storage',
    firestore: { host: '127.0.0.1', port: 8087 },
    storage: { host: '127.0.0.1', port: 9199, rules: await readFile(new URL('../storage.rules', import.meta.url), 'utf8') } });
  await env.withSecurityRulesDisabled(async (ctx) => {
    await setDoc(doc(ctx.firestore(), 'robots/robot'), { enabled: true, senderId: 'sender', viewerUids: ['viewer'] });
    await setDoc(doc(ctx.firestore(), 'device_uploaders/sender'), { enabled: true });
    await setDoc(doc(ctx.firestore(), 'device_uploaders/other-sender'), { enabled: true });
  });
});
after(async () => { await env?.cleanup(); });

test('live image upload and download require the assigned robot identities', async () => {
  await assertSucceeds(uploadBytes(ref(storage('sender'), object), image, { contentType: 'image/jpeg' }));
  await assertSucceeds(getBytes(ref(storage('viewer', admin), object)));
  await assertFails(getBytes(ref(storage('outsider', admin), object)));
  await assertFails(getBytes(ref(env.unauthenticatedContext().storage(), object)));
  await assertFails(uploadBytes(ref(storage('other-sender'), object), image, { contentType: 'image/jpeg' }));
  await assertFails(uploadBytes(ref(storage('viewer', admin), object), image, { contentType: 'image/jpeg' }));
  await assertFails(uploadBytes(ref(storage('sender'), object), image, { contentType: 'text/plain' }));
  await assertFails(deleteObject(ref(storage('sender'), object)));
  await assertFails(deleteObject(ref(storage('viewer', admin), object)));
});

test('legacy administrator paths remain accessible without bypassing robot scope', async () => {
  const legacy = ref(storage('viewer', admin), 'dashboard/media/test.jpg');
  await assertSucceeds(uploadBytes(legacy, image, { contentType: 'image/jpeg' }));
  await assertSucceeds(getBytes(legacy));
  await assertSucceeds(deleteObject(legacy));
});
