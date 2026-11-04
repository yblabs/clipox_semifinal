// ClipOX edge proxy.
//
// The static site is served from Workers assets; everything under /api,
// /media and /fonts belongs to the FastAPI backend, which is exposed from
// this machine through a Cloudflare Quick Tunnel. The tunnel URL lives in
// the API_ORIGIN secret so it can be rotated (every tunnel restart mints a
// new URL) without redeploying the Worker.
const PROXIED_PREFIXES = ["/api", "/media", "/fonts"];

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;
    const proxied = PROXIED_PREFIXES.some(
      (p) => path === p || path.startsWith(p + "/")
    );

    if (!proxied) {
      // The assets router serves every matching file before this Worker runs
      // (pretty URLs and .html redirects included), so reaching here means no
      // asset matched: return the same 404 the assets-only setup produced.
      return new Response("Not Found", {
        status: 404,
        headers: { "content-type": "text/plain; charset=utf-8" },
      });
    }

    const origin = (env.API_ORIGIN || "").replace(/\/+$/, "");
    if (!origin) {
      return json({ error: "backend_origin_unconfigured" }, 503);
    }

    try {
      const target = origin + path + url.search;
      return await fetch(new Request(target, request));
    } catch (err) {
      return json(
        {
          error: "backend_unreachable",
          detail: String((err && err.message) || err),
        },
        502
      );
    }
  },
};

function json(obj, status) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });
}
