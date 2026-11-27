// スマホのアプリとして使うための仕組み（サービスワーカー）。
// - 画面と判定データは「まず通信して最新を取り、取れなければ前回保存した内容を出す」（電波が弱くても開ける）
// - アイコンなど変わらないものは保存した内容をそのまま使う
// 画面の作りを大きく変えたら VERSION を上げる（古い保存内容を消して入れ替える）
const VERSION = "v2";
const CACHE = `swing-${VERSION}`;
const SHELL = ["./", "index.html", "manifest.webmanifest", "icons/icon-192.png", "icons/icon-512.png", "icons/apple-touch-icon.png"];

self.addEventListener("install", ev => {
  ev.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", ev => {
  ev.waitUntil(caches.keys()
    .then(keys => Promise.all(keys.filter(k => k.startsWith("swing-") && k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});

// 通信できずに保存済みの内容を返したことを、画面に知らせる印（x-from-cache）を付ける
function marked(r) {
  const h = new Headers(r.headers);
  h.set("x-from-cache", "1");
  return new Response(r.body, {status: r.status, statusText: r.statusText, headers: h});
}

self.addEventListener("fetch", ev => {
  const req = ev.request;
  const url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== location.origin) return;   // 外部サイト（株探・フォントなど）は中継しない
  const key = url.origin + url.pathname;                                 // ?t=… の違いでデータが溜まらないように
  const fresh = url.pathname.endsWith("/") || url.pathname.endsWith(".html") || url.pathname.includes("/data/");
  if (fresh) {
    ev.respondWith(fetch(req).then(res => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(key, copy)); }
      return res;
    }).catch(() => caches.match(key).then(r => r || caches.match(url.pathname.endsWith("/") ? "index.html" : key))
      .then(r => r ? marked(r) : Response.error())));
  } else {
    ev.respondWith(caches.match(key).then(r => r || fetch(req).then(res => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(key, copy)); }
      return res;
    })));
  }
});
