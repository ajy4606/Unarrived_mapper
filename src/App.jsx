import { useState, useRef, useEffect, useCallback } from "react";
import * as THREE from "three";
import { pipeline } from "@huggingface/transformers";
import JSZip from "jszip";

/* ═══════════════════════════════════════════════════════════════
   AI SINGLETONS
═══════════════════════════════════════════════════════════════ */
let _depthP = null,
  _segP = null;
const getDepth = () => {
  if (!_depthP)
    _depthP = pipeline(
      "depth-estimation",
      "onnx-community/depth-anything-v2-small",
      {
        // ✅ 올바른 주소
        device: "wasm", // WebGPU 호환성 에러 방지를 위해 wasm으로 변경
      },
    ).catch((e) => {
      _depthP = null;
      throw e;
    });
  return _depthP;
};
const getSeg = () => {
  if (!_segP)
    _segP = pipeline(
      "image-segmentation",
      "Xenova/segformer-b0-finetuned-ade-512-512",
    ).catch((e) => {
      _segP = null;
      throw e;
    });
  return _segP;
};

/* ═══════════════════════════════════════════════════════════════
   GPU QUERY
═══════════════════════════════════════════════════════════════ */
function queryGPU() {
  try {
    const c = document.createElement("canvas"),
      gl = c.getContext("webgl2");
    if (!gl) return { maxTex: 2048, renderer: "WebGL2 N/A" };
    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const renderer = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : "GPU";
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return { maxTex, renderer };
  } catch {
    return { maxTex: 2048, renderer: "N/A" };
  }
}
const vramMB = (res) => Math.round((res * res * 4 * 12) / 1024 / 1024);

/* ═══════════════════════════════════════════════════════════════
   PROCEDURAL HDRI
═══════════════════════════════════════════════════════════════ */
const HDRI = {
  studio: {
    sky: ["#0d1020", "#1a2040", "#404860", "#707890"],
    key: { color: "#ffe8c8", x: 140, y: 55, r: 90 },
    fill: { color: "#c8d8ff", x: 380, y: 100, r: 60 },
    ground: "#181020",
  },
  outdoor: {
    sky: ["#0a1a40", "#1a4090", "#4090d0", "#90c8f0"],
    key: { color: "#fff4d0", x: 100, y: 40, r: 110 },
    fill: { color: "#80b0e0", x: 450, y: 80, r: 80 },
    ground: "#2a3018",
  },
  warm: {
    sky: ["#100810", "#302030", "#704830", "#c07040"],
    key: { color: "#ffd090", x: 160, y: 50, r: 100 },
    fill: { color: "#ff8040", x: 400, y: 90, r: 55 },
    ground: "#201008",
  },
};
function buildEnv(renderer, key) {
  const p = HDRI[key];
  const W = 1024,
    H = 512;
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const ctx = c.getContext("2d");
  const sg = ctx.createLinearGradient(0, 0, 0, H);
  p.sky.forEach((col, i) => sg.addColorStop(i / (p.sky.length - 1), col));
  ctx.fillStyle = sg;
  ctx.fillRect(0, 0, W, H);
  const gg = ctx.createLinearGradient(0, H * 0.65, 0, H);
  gg.addColorStop(0, "rgba(0,0,0,0)");
  gg.addColorStop(1, p.ground);
  ctx.fillStyle = gg;
  ctx.fillRect(0, 0, W, H);
  [p.key, p.fill].forEach(({ color, x, y, r }) => {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, color + "ee");
    g.addColorStop(0.4, color + "55");
    g.addColorStop(1, color + "00");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
  });
  const tex = new THREE.CanvasTexture(c);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.SRGBColorSpace; // 텍스처 에러 수정 부분
  const pm = new THREE.PMREMGenerator(renderer);
  pm.compileEquirectangularShader();
  const env = pm.fromEquirectangular(tex).texture;
  pm.dispose();
  tex.dispose();
  return env;
}

/* ═══════════════════════════════════════════════════════════════
   MATERIAL CLASS (ADE20k → 0..4)
═══════════════════════════════════════════════════════════════ */
const LABEL_MAP = {
  car: 1,
  bus: 1,
  truck: 1,
  metal: 1,
  steel: 1,
  iron: 1,
  wall: 2,
  floor: 2,
  rock: 2,
  stone: 2,
  concrete: 2,
  building: 2,
  curtain: 3,
  sofa: 3,
  bed: 3,
  fabric: 3,
  cloth: 3,
  rug: 3,
  bottle: 4,
  glass: 4,
  window: 4,
  plastic: 4,
  jar: 4,
};
const labelCls = (l) => {
  const low = l.toLowerCase();
  for (const [k, v] of Object.entries(LABEL_MAP)) if (low.includes(k)) return v;
  return 0;
};

async function buildMatMask(segs) {
  const S = 512,
    c = document.createElement("canvas");
  c.width = c.height = S;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  const id = ctx.createImageData(S, S);
  id.data.fill(0);
  for (let i = 3; i < id.data.length; i += 4) id.data[i] = 255;
  for (const seg of segs || []) {
    if (!seg?.mask) continue;
    const cls = labelCls(seg.label || "");
    const enc = Math.round((cls / 4) * 255);
    try {
      const mc = seg.mask.toCanvas?.();
      if (!mc) continue;
      const tc = document.createElement("canvas");
      tc.width = tc.height = S;
      tc.getContext("2d").drawImage(mc, 0, 0, S, S);
      const md = tc.getContext("2d").getImageData(0, 0, S, S).data;
      for (let i = 0; i < id.data.length; i += 4)
        if (md[i] > 128) {
          id.data[i] = enc;
          id.data[i + 1] = 0;
          id.data[i + 2] = 0;
        }
    } catch {}
  }
  ctx.putImageData(id, 0, 0);
  return c;
}

/* ═══════════════════════════════════════════════════════════════
   GLSL3 SHADERS
═══════════════════════════════════════════════════════════════ */
const VS = `#version 300 es\nin vec2 a;out vec2 v;\nvoid main(){v=a*.5+.5;gl_Position=vec4(a,0.,1.);}`;
const FS_PASS = `#version 300 es\nprecision highp float;\nuniform sampler2D T;in vec2 v;out vec4 o;\nvoid main(){o=texture(T,v);}`;
const FS_BH = `#version 300 es\nprecision highp float;\nuniform sampler2D T;uniform vec2 px;uniform float radius;in vec2 v;out vec4 o;const float W[5]=float[](0.227027,0.194595,0.121622,0.054054,0.016216);void main(){vec4 c=texture(T,v)*W[0];for(int i=1;i<5;i++)c+=texture(T,v+vec2(float(i)*px.x*radius,0.))*W[i]+texture(T,v-vec2(float(i)*px.x*radius,0.))*W[i];o=c;}`;
const FS_BV = `#version 300 es\nprecision highp float;\nuniform sampler2D T;uniform vec2 px;uniform float radius;in vec2 v;out vec4 o;const float W[5]=float[](0.227027,0.194595,0.121622,0.054054,0.016216);void main(){vec4 c=texture(T,v)*W[0];for(int i=1;i<5;i++)c+=texture(T,v+vec2(0.,float(i)*px.y*radius))*W[i]+texture(T,v-vec2(0.,float(i)*px.y*radius))*W[i];o=c;}`;
const FS_HP = `#version 300 es\nprecision highp float;\nuniform sampler2D ORIG,BLUR;in vec2 v;out vec4 o;void main(){o=clamp(texture(ORIG,v)-texture(BLUR,v)+vec4(.5),0.,1.);}`;
const FS_NM = `#version 300 es\nprecision highp float;
uniform sampler2D DEPTH,DETAIL;uniform vec2 px;uniform float str,detailMix,sigma;
in vec2 v;out vec4 o;
float bilW(float dc,float ds){float d=dc-ds;return exp(-d*d/(sigma*sigma+.0001));}
float hD(vec2 u){return texture(DEPTH,u).r;}
float hA(vec2 u){return texture(DETAIL,u).r-.5;}
void main(){
  float dc=hD(v);
  vec2 ofs[8];ofs[0]=px*vec2(-1,1);ofs[1]=px*vec2(0,1);ofs[2]=px*vec2(1,1);
  ofs[3]=px*vec2(-1,0);ofs[4]=px*vec2(1,0);ofs[5]=px*vec2(-1,-1);ofs[6]=px*vec2(0,-1);ofs[7]=px*vec2(1,-1);
  float kx[8];kx[0]=-1.;kx[1]=0.;kx[2]=1.;kx[3]=-2.;kx[4]=2.;kx[5]=-1.;kx[6]=0.;kx[7]=1.;
  float ky[8];ky[0]=1.;ky[1]=2.;ky[2]=1.;ky[3]=0.;ky[4]=0.;ky[5]=-1.;ky[6]=-2.;ky[7]=-1.;
  float dx=0.,dy=0.,adx=0.,ady=0.;
  for(int i=0;i<8;i++){float ds=hD(v+ofs[i]);float bw=bilW(dc,ds);float as=hA(v+ofs[i]);dx+=kx[i]*ds*bw;dy+=ky[i]*ds*bw;adx+=kx[i]*as;ady+=ky[i]*as;}
  vec3 nD=normalize(vec3(dx*str,dy*str,1.));vec3 nA=normalize(vec3(adx*str*1.5,ady*str*1.5,1.));
  vec3 n=normalize(vec3(nD.xy+nA.xy*detailMix,nD.z));o=vec4(n*.5+.5,1.);}`;
const FS_DISP = `#version 300 es\nprecision highp float;\nuniform sampler2D T;uniform float mid,con,inv;in vec2 v;out vec4 o;void main(){float d=texture(T,v).r;d=clamp((d-mid)*con+.5,0.,1.);d=mix(d,1.-d,inv);o=vec4(d,d,d,1.);}`;
const FS_ROUGH = `#version 300 es\nprecision highp float;
uniform sampler2D DEPTH,DETAIL,SEG;uniform vec2 px;uniform float con,bias,inv,detailMix,matBlend;
in vec2 v;out vec4 o;
void main(){
  float sd=0.,sd2=0.;for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++){float s=texture(DEPTH,v+px*vec2(float(dx),float(dy))).r;sd+=s;sd2+=s*s;}
  sd/=25.;sd2/=25.;float lv=max(sd2-sd*sd,0.)*80.;
  float fe=length(texture(DETAIL,v).rgb-vec3(.5))*2.;
  float lb=texture(DEPTH,v).r;float l=mix(lb,mix(lv,fe,.5),detailMix);
  float cls=texture(SEG,v).r*4.;
  float rp=.75;rp=mix(rp,.38,step(.5,cls)*step(cls,1.5));rp=mix(rp,.86,step(1.5,cls)*step(cls,2.5));rp=mix(rp,.92,step(2.5,cls)*step(cls,3.5));rp=mix(rp,.22,step(3.5,cls));
  l=mix(l,rp,matBlend);l=mix(l,1.-l,inv);o=vec4(vec3(clamp((l-.5)*con+.5+bias,0.,1.)),1.);}`;
const FS_METAL = `#version 300 es\nprecision highp float;\nuniform sampler2D T,SEG;uniform float thr,soft;in vec2 v;out vec4 o;void main(){float l=texture(T,v).r;float cls=texture(SEG,v).r*4.;float bias=0.;bias=mix(bias,-.28,step(.5,cls)*step(cls,1.5));bias=mix(bias,.18,step(3.5,cls));float m=smoothstep(thr+bias-soft,thr+bias+soft,l);o=vec4(m,m,m,1.);}`;
const FS_AO = `#version 300 es\nprecision highp float;\nuniform sampler2D T;uniform vec2 px;uniform float radius,strength;in vec2 v;out vec4 o;const float G=2.3999632;const int S=32;void main(){float dc=texture(T,v).r,occ=0.,dMn=1.,dMx=0.;for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++){float s=texture(T,v+px*vec2(float(dx),float(dy))*4.).r;dMn=min(dMn,s);dMx=max(dMx,s);}float dr=max(dMx-dMn,.01),ar=radius/dr;for(int i=0;i<S;i++){float r=sqrt(float(i+1)/float(S)),th=float(i)*G;vec2 off=vec2(cos(th),sin(th))*px*min(ar,radius)*80.*r;occ+=clamp((dc-texture(T,v+off).r)*8.,0.,1.)*(1.-r)*.7+.3*(1.-r);}float ao=clamp(1.-(occ/float(S)*2.)*strength,0.,1.);o=vec4(ao,ao,ao,1.);}`;
const FS_CURV = `#version 300 es\nprecision highp float;\nuniform sampler2D NM;uniform vec2 px;uniform float scale;in vec2 v;out vec4 o;float cs(float s){vec3 n=normalize(texture(NM,v).rgb*2.-1.);float d=0.;vec2[4] ds=vec2[4](vec2(px.x*s,0.),vec2(0.,px.y*s),vec2(-px.x*s,0.),vec2(0.,-px.y*s));for(int i=0;i<4;i++)d+=dot(n,normalize(texture(NM,v+ds[i]).rgb*2.-1.));return clamp(-(d/4.-1.)*scale*3.+.5,0.,1.);}void main(){float c=cs(1.)*.5+cs(4.)*.3+cs(12.)*.2;o=vec4(c,c,c,1.);}`;
const FS_ORM = `#version 300 es\nprecision highp float;\nuniform sampler2D AO,RO,ME;in vec2 v;out vec4 o;void main(){o=vec4(texture(AO,v).r,texture(RO,v).r,texture(ME,v).r,1.);}`;

/* ═══════════════════════════════════════════════════════════════
   WebGL2 PIPELINE
═══════════════════════════════════════════════════════════════ */
const TILE = 2048,
  PAD = 32;
function mkGL(W, H) {
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const gl = c.getContext("webgl2", { preserveDrawingBuffer: true });
  if (!gl) return null;
  const mp = (fs) => {
    const p = gl.createProgram();
    [
      [gl.VERTEX_SHADER, VS],
      [gl.FRAGMENT_SHADER, fs],
    ].forEach(([t, s]) => {
      const sh = gl.createShader(t);
      gl.shaderSource(sh, s);
      gl.compileShader(sh);
      gl.attachShader(p, sh);
    });
    gl.linkProgram(p);
    return p;
  };
  const P = {
    pass: mp(FS_PASS),
    bh: mp(FS_BH),
    bv: mp(FS_BV),
    hp: mp(FS_HP),
    nm: mp(FS_NM),
    disp: mp(FS_DISP),
    rough: mp(FS_ROUGH),
    metal: mp(FS_METAL),
    ao: mp(FS_AO),
    curv: mp(FS_CURV),
    orm: mp(FS_ORM),
  };
  const qb = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, qb);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
    gl.STATIC_DRAW,
  );
  const mkt = (src) => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    if (src) {
      // ✅ FIX: Canvas Y축(상단=0)과 WebGL 텍스처 Y축(하단=0)이 반대
      // UNPACK_FLIP_Y_WEBGL=true로 업로드 시 수직 반전 → 올바른 방향으로 렌더링
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    } else {
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        W,
        H,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        null,
      );
    }
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  };
  const upt = (t, s) => {
    // ✅ FIX: 동적 업로드에도 동일한 Y축 보정 적용
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, s);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  };
  const mkf = (t) => {
    const f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      t,
      0,
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return f;
  };
  const exec = (prog, fbo, binds, uns) => {
    gl.useProgram(prog);
    const a = gl.getAttribLocation(prog, "a");
    gl.enableVertexAttribArray(a);
    gl.bindBuffer(gl.ARRAY_BUFFER, qb);
    gl.vertexAttribPointer(a, 2, gl.FLOAT, false, 0, 0);
    binds.forEach(([n, t], i) => {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.uniform1i(gl.getUniformLocation(prog, n), i);
    });
    for (const [k, v] of Object.entries(uns)) {
      const u = gl.getUniformLocation(prog, k);
      if (u === null) continue; // ✅ FIX: location=0은 유효 — !u 는 0을 falsy로 처리해 스킵하는 버그
      Array.isArray(v) ? gl.uniform2fv(u, v) : gl.uniform1f(u, v);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo ?? null);
    gl.viewport(0, 0, W, H);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  };
  const T = {
    a: mkt(null),
    d: mkt(null),
    s: mkt(null),
    bh: mkt(null),
    bv: mkt(null),
    hi: mkt(null),
    nm: mkt(null),
    ao: mkt(null),
    ro: mkt(null),
    me: mkt(null),
    di: mkt(null),
  };
  const F = {};
  for (const k of Object.keys(T)) F[k] = mkf(T[k]);
  return { gl, c, P, T, F, exec, upt };
}
function rgba(src, W, H) {
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  c.getContext("2d", { willReadFrequently: true }).drawImage(src, 0, 0, W, H);
  return c;
}
function runTile(ctx, albC, depC, segC, W, H, s) {
  const { gl, c, P, T, F, exec, upt } = ctx;
  const px = [1 / W, 1 / H];
  upt(T.a, albC);
  upt(T.d, depC);
  upt(T.s, segC);
  exec(P.bh, F.bh, [["T", T.a]], { px, radius: s.blurRadius });
  exec(P.bv, F.bv, [["T", T.bh]], { px, radius: s.blurRadius });
  exec(
    P.hp,
    F.hi,
    [
      ["ORIG", T.a],
      ["BLUR", T.bv],
    ],
    {},
  );
  exec(
    P.nm,
    F.nm,
    [
      ["DEPTH", T.d],
      ["DETAIL", T.hi],
    ],
    { px, str: s.normalStr, detailMix: s.detailMix, sigma: s.normalSigma },
  );
  exec(P.ao, F.ao, [["T", T.d]], { px, radius: s.aoRadius, strength: s.aoStr });
  exec(
    P.rough,
    F.ro,
    [
      ["DEPTH", T.d],
      ["DETAIL", T.hi],
      ["SEG", T.s],
    ],
    {
      px,
      con: s.roughCon,
      bias: s.roughBias,
      inv: s.roughInv ? 1 : 0,
      detailMix: s.roughDetailMix,
      matBlend: s.matBlend,
    },
  );
  exec(
    P.metal,
    F.me,
    [
      ["T", T.d],
      ["SEG", T.s],
    ],
    { thr: s.metalThr, soft: s.metalSoft },
  );
  exec(P.disp, F.di, [["T", T.d]], {
    mid: s.dispMid,
    con: s.dispCon,
    inv: s.dispInv ? 1 : 0,
  });
  const snap = (prog, binds, uns) => {
    exec(prog, null, binds, uns);
    return c.toDataURL("image/png");
  };
  return {
    normal: snap(P.pass, [["T", T.nm]], {}),
    displacement: snap(P.disp, [["T", T.d]], {
      mid: s.dispMid,
      con: s.dispCon,
      inv: s.dispInv ? 1 : 0,
    }),
    roughness: snap(P.pass, [["T", T.ro]], {}),
    metallic: snap(P.pass, [["T", T.me]], {}),
    ao: snap(P.pass, [["T", T.ao]], {}),
    curvature: snap(P.curv, [["NM", T.nm]], { px, scale: s.curvScale }),
    orm: snap(
      P.orm,
      [
        ["AO", T.ao],
        ["RO", T.ro],
        ["ME", T.me],
      ],
      {},
    ),
  };
}
async function generateMaps(img, depCvs, segCvs, s, res, onProg) {
  const sc = Math.min(1, res / Math.max(img.naturalWidth, img.naturalHeight));
  const W = Math.round(img.naturalWidth * sc),
    H = Math.round(img.naturalHeight * sc);
  const neutral =
    segCvs ||
    (() => {
      const c = document.createElement("canvas");
      c.width = c.height = 64;
      c.getContext("2d").fillRect(0, 0, 64, 64);
      return c;
    })();
  if (W <= TILE && H <= TILE) {
    const ctx = mkGL(W, H);
    if (!ctx) return null;
    onProg?.(1, 1);
    return runTile(
      ctx,
      rgba(img, W, H),
      rgba(depCvs, W, H),
      rgba(neutral, W, H),
      W,
      H,
      s,
    );
  }
  const TW = TILE + 2 * PAD,
    TH = TILE + 2 * PAD;
  const ctx = mkGL(TW, TH);
  if (!ctx) return null;
  const txA = Math.ceil(W / TILE),
    tyA = Math.ceil(H / TILE),
    tot = txA * tyA;
  const fAlb = rgba(img, W, H),
    fDep = rgba(depCvs, W, H),
    fSeg = rgba(neutral, W, H);
  const keys = [
    "normal",
    "displacement",
    "roughness",
    "metallic",
    "ao",
    "curvature",
    "orm",
  ];
  const outs = {};
  for (const k of keys) {
    const c = document.createElement("canvas");
    c.width = W;
    c.height = H;
    outs[k] = c;
  }
  const tA = document.createElement("canvas");
  tA.width = tA.height = TW;
  const tD = document.createElement("canvas");
  tD.width = tD.height = TH;
  const tS = document.createElement("canvas");
  tS.width = tS.height = TH;
  let tn = 0;
  for (let ty = 0; ty < tyA; ty++)
    for (let tx = 0; tx < txA; tx++) {
      const ox = -(tx * TILE - PAD),
        oy = -(ty * TILE - PAD);
      for (const [tc, src] of [
        [tA, fAlb],
        [tD, fDep],
        [tS, fSeg],
      ]) {
        const c = tc.getContext("2d", { willReadFrequently: true });
        c.fillStyle = "#808080";
        c.fillRect(0, 0, TW, TH);
        c.drawImage(src, ox, oy);
      }
      const r = runTile(ctx, tA, tD, tS, TW, TH, s);
      const cW = Math.min(TILE, W - tx * TILE),
        cH = Math.min(TILE, H - ty * TILE),
        dX = tx * TILE,
        dY = ty * TILE;
      for (const k of keys) {
        const ti = new Image();
        ti.src = r[k];
        await new Promise((r) => {
          ti.onload = r;
          ti.onerror = r;
        });
        outs[k]
          .getContext("2d")
          .drawImage(ti, PAD, PAD, cW, cH, dX, dY, cW, cH);
      }
      tn++;
      onProg?.(tn, tot);
      await new Promise((r) => setTimeout(r, 0));
    }
  const out = {};
  for (const k of keys) out[k] = outs[k].toDataURL("image/png");
  return out;
}

/* ═══════════════════════════════════════════════════════════════
   AUTO-ANALYZE
═══════════════════════════════════════════════════════════════ */
function analyzeAlbedo(img) {
  const S = 128,
    c = document.createElement("canvas");
  c.width = c.height = S;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, S, S);
  const px = ctx.getImageData(0, 0, S, S).data,
    n = px.length / 4;
  let ss = 0,
    sl = 0,
    la = [];
  for (let i = 0; i < px.length; i += 4) {
    const r = px[i] / 255,
      g = px[i + 1] / 255,
      b = px[i + 2] / 255;
    const mx = Math.max(r, g, b),
      mn = Math.min(r, g, b),
      l = 0.299 * r + 0.587 * g + 0.114 * b;
    ss += mx === 0 ? 0 : (mx - mn) / mx;
    sl += l;
    la.push(l);
  }
  const aS = ss / n,
    aL = sl / n,
    lv = la.reduce((a, l) => a + (l - aL) ** 2, 0) / n;
  return {
    metalThr: aS < 0.12 ? 0.52 : aS < 0.25 ? 0.68 : 0.84,
    roughCon: Math.min(5.5, 1.5 + lv * 40),
    roughBias: lv > 0.04 ? -0.05 : 0.05,
    detailMix: Math.min(0.8, 0.2 + lv * 18),
    roughDetailMix: Math.min(0.8, 0.3 + lv * 14),
    dispMid: Math.max(0.3, Math.min(0.7, aL)),
    matBlend: Math.min(0.5, aS < 0.15 ? 0.4 : 0.15),
  };
}

/* ═══════════════════════════════════════════════════════════════
   ZIP
═══════════════════════════════════════════════════════════════ */
async function doZip(maps, res) {
  const z = new JSZip(),
    f = z.folder(`unarrived_${res}px`);
  await Promise.all(
    Object.entries(maps).map(async ([k, u]) => {
      f.file(`${k}.png`, await (await fetch(u)).blob());
    }),
  );
  const b = await z.generateAsync({ type: "blob" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b);
  a.download = `unarrived_${res}px.zip`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ═══════════════════════════════════════════════════════════════
   DESIGN TOKENS
═══════════════════════════════════════════════════════════════ */
const C = {
  bg: "#0e0f14",
  sidebar: "#13141a",
  panel: "#1a1b22",
  border: "#2a2b38",
  borderHi: "#3a3b50",
  text: "#c8cad8",
  textDim: "#6a6c80",
  textFaint: "#3a3c50",
  accent: "#e8a020",
  accentDim: "#a06010",
  blue: "#5090e8",
  green: "#40c060",
  purple: "#9060d0",
  teal: "#30b898",
  red: "#c04030",
  yellow: "#d0b020",
  pink: "#c04090",
};

const MAPS_DEF = [
  {
    key: "normal",
    label: "Normal Map",
    color: C.blue,
    hint: "Bilateral depth + highpass blend",
  },
  {
    key: "displacement",
    label: "Displacement",
    color: C.accent,
    hint: "AI depth height field",
  },
  {
    key: "roughness",
    label: "Roughness",
    color: C.purple,
    hint: "Frequency + material class",
  },
  {
    key: "metallic",
    label: "Metallic",
    color: C.teal,
    hint: "Material-aware threshold",
  },
  {
    key: "ao",
    label: "Ambient Occ.",
    color: C.red,
    hint: "HBAO 32-sample spiral",
  },
  {
    key: "curvature",
    label: "Curvature",
    color: C.yellow,
    hint: "Multi-scale edge wear",
  },
];
const MAPS_EXTRA = [
  {
    key: "orm",
    label: "ORM Pack",
    color: C.blue,
    hint: "R:AO  G:Rough  B:Metal  (UE/Unity)",
  },
];
const ALL_MAPS = [...MAPS_DEF, ...MAPS_EXTRA];

const STEPS = {
  idle: { label: "Idle", dot: C.textFaint },
  loading: { label: "Loading depth model…", dot: C.blue },
  inferring: { label: "Running depth AI…", dot: C.green },
  segload: { label: "Loading segmenter…", dot: C.purple },
  seginfer: { label: "Analyzing materials…", dot: C.pink },
  segmask: { label: "Building material mask…", dot: C.pink },
  generating: { label: "Generating PBR maps…", dot: C.accent },
  tiling: { label: "Tiling…", dot: C.accent },
  analyzing: { label: "Analyzing albedo…", dot: C.pink },
  ready: { label: "Ready", dot: C.green },
  error: { label: "Error — check console", dot: C.red },
};

const SHAPES = [
  { k: "sphere", l: "Sphere" },
  { k: "plane", l: "Plane" },
  { k: "torus", l: "Torus" },
];
const ALL_RES = [512, 1024, 2048, 4096, 8192];

/* ═══════════════════════════════════════════════════════════════
   APP
═══════════════════════════════════════════════════════════════ */
const DEF = {
  normalStr: 3,
  detailMix: 0.4,
  normalSigma: 0.05,
  dispMid: 0.5,
  dispCon: 2,
  dispInv: false,
  roughCon: 2,
  roughBias: 0,
  roughInv: false,
  roughDetailMix: 0.5,
  metalThr: 0.75,
  metalSoft: 0.1,
  blurRadius: 6,
  aoRadius: 0.8,
  aoStr: 1.2,
  curvScale: 1,
  matBlend: 0.25,
};

export default function App() {
  const [srcURL, setSrcURL] = useState(null);
  const [maps, setMaps] = useState(null);
  const [settings, setSettings] = useState(DEF);
  const [proc, setProc] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [hovered, setHovered] = useState(null);
  const [imgInfo, setImgInfo] = useState(null);
  const [aiStep, setAiStep] = useState("idle");
  const [tileProg, setTileProg] = useState(null);
  const [quality, setQuality] = useState(false);
  const [shape, setShape] = useState("sphere");
  const [tileRep, setTileRep] = useState(1);
  const [res, setRes] = useState(1024);
  const [hdri, setHdri] = useState("studio");
  const [gpu, setGpu] = useState({ maxTex: 4096, renderer: "Querying…" });
  const [hint, setHint] = useState(null);

  const cvs = useRef(null),
    imgR = useRef(null),
    depR = useRef(null),
    segR = useRef(null);
  const setR = useRef(settings),
    resR = useRef(res),
    threeR = useRef({}),
    orbit = useRef({ on: false, x: 0, y: 0 }),
    tok = useRef(0);
  useEffect(() => {
    setR.current = settings;
  }, [settings]);
  useEffect(() => {
    resR.current = res;
  }, [res]);
  useEffect(() => {
    setGpu(queryGPU());
  }, []);

  const loadFile = useCallback(
    (file) => {
      if (!file?.type.startsWith("image/")) return;
      const tid = ++tok.current,
        url = URL.createObjectURL(file),
        img = new Image();
      img.onload = async () => {
        if (tid !== tok.current) return;
        imgR.current = img;
        setSrcURL(url);
        setImgInfo({ w: img.naturalWidth, h: img.naturalHeight });
        setMaps(null);
        depR.current = null;
        segR.current = null;
        setHint(null);
        setTileProg(null);
        setProc(true);
        try {
          setAiStep("loading");
          const dep = await getDepth();
          if (tid !== tok.current) return;
          setAiStep("inferring");
          const dr = await dep(url);
          if (tid !== tok.current) return;
          depR.current = dr.depth.toCanvas();
          if (quality) {
            setAiStep("segload");
            const sg = await getSeg();
            if (tid !== tok.current) return;
            setAiStep("seginfer");
            const sr = await sg(url);
            if (tid !== tok.current) return;
            setAiStep("segmask");
            segR.current = await buildMatMask(sr);
          }
          setAiStep("generating");
          await new Promise((r) => requestAnimationFrame(r));
          if (tid !== tok.current) return;
          const isTile =
            Math.min(
              1,
              resR.current / Math.max(img.naturalWidth, img.naturalHeight),
            ) *
              Math.max(img.naturalWidth, img.naturalHeight) >
            TILE;
          if (isTile) setAiStep("tiling");
          const result = await generateMaps(
            imgR.current,
            depR.current,
            segR.current,
            setR.current,
            resR.current,
            (c, t) => {
              if (tid === tok.current) setTileProg({ c, t });
            },
          );
          if (tid !== tok.current) return;
          setMaps(result);
          setAiStep("ready");
          setTileProg(null);
        } catch (err) {
          if (tid !== tok.current) return;
          console.error(err);
          setAiStep("error");
        } finally {
          if (tid === tok.current) setProc(false);
        }
      };
      img.src = url;
    },
    [quality],
  );

  useEffect(() => {
    if (!depR.current || !imgR.current) return;
    const id = setTimeout(async () => {
      setProc(true);
      setTileProg(null);
      const isTile =
        Math.min(
          1,
          res / Math.max(imgR.current.naturalWidth, imgR.current.naturalHeight),
        ) *
          Math.max(imgR.current.naturalWidth, imgR.current.naturalHeight) >
        TILE;
      setAiStep(isTile ? "tiling" : "generating");
      const r = await generateMaps(
        imgR.current,
        depR.current,
        segR.current,
        settings,
        res,
        (c, t) => setTileProg({ c, t }),
      );
      setMaps(r);
      setAiStep("ready");
      setTileProg(null);
      setProc(false);
    }, 250);
    return () => clearTimeout(id);
  }, [settings, res]);

  /* Three.js */
  useEffect(() => {
    const el = cvs.current;
    if (!el) return;
    const renderer = new THREE.WebGLRenderer({ canvas: el, antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    renderer.outputColorSpace = THREE.SRGBColorSpace; // 텍스처 에러 수정 부분
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 50);
    camera.position.z = 3.2;
    const env = buildEnv(renderer, "studio");
    scene.environment = env;
    scene.background = env;

    // 조명 에러 수정 부분
    const light = new THREE.DirectionalLight(0xfff4d6, 0.6);
    light.position.set(4, 5, 3);
    scene.add(light);

    const G = {
      sphere: new THREE.SphereGeometry(1, 128, 64),
      plane: new THREE.PlaneGeometry(2, 2, 64, 64),
      torus: new THREE.TorusGeometry(0.72, 0.34, 64, 128),
    };
    Object.values(G).forEach((g) => g.setAttribute("uv2", g.attributes.uv));
    const mat = new THREE.MeshStandardMaterial({
      roughness: 0.5,
      metalness: 0,
      color: 0xd0d0d0,
      envMapIntensity: 1,
    });
    const mesh = new THREE.Mesh(G.sphere, mat);
    scene.add(mesh);
    const gm = new THREE.Mesh(
      new THREE.PlaneGeometry(20, 20),
      new THREE.MeshStandardMaterial({ color: 0x0a0a18, roughness: 1 }),
    );
    gm.rotation.x = -Math.PI / 2;
    gm.position.y = -1.4;
    scene.add(gm);
    threeR.current = { renderer, scene, camera, mat, mesh, G };
    const resize = () => {
      const w = el.clientWidth,
        h = el.clientHeight || 1;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    let raf;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      if (!orbit.current.on) mesh.rotation.y += 0.005;
      renderer.render(scene, camera);
    };
    tick();
    const od = (e) => {
      orbit.current = { on: true, x: e.clientX, y: e.clientY };
      el.style.cursor = "grabbing";
    };
    const ou = () => {
      orbit.current.on = false;
      el.style.cursor = "grab";
    };
    const om = (e) => {
      if (!orbit.current.on) return;
      mesh.rotation.y += (e.clientX - orbit.current.x) * 0.009;
      mesh.rotation.x = Math.max(
        -1.1,
        Math.min(1.1, mesh.rotation.x + (e.clientY - orbit.current.y) * 0.009),
      );
      orbit.current.x = e.clientX;
      orbit.current.y = e.clientY;
    };
    el.addEventListener("mousedown", od);
    window.addEventListener("mouseup", ou);
    window.addEventListener("mousemove", om);
    return () => {
      cancelAnimationFrame(raf);
      renderer.dispose();
      ro.disconnect();
      el.removeEventListener("mousedown", od);
      window.removeEventListener("mouseup", ou);
      window.removeEventListener("mousemove", om);
    };
  }, []);
  useEffect(() => {
    const { renderer, scene } = threeR.current;
    if (!renderer || !scene) return;
    const e = buildEnv(renderer, hdri);
    scene.environment = e;
    scene.background = e;
  }, [hdri]);
  useEffect(() => {
    const { mesh, G } = threeR.current;
    if (!mesh) return;
    mesh.geometry = G[shape] ?? G.sphere;
  }, [shape]);
  useEffect(() => {
    const { mat } = threeR.current;
    if (!mat) return;
    [
      mat.map,
      mat.normalMap,
      mat.roughnessMap,
      mat.metalnessMap,
      mat.aoMap,
    ].forEach((t) => {
      if (!t) return;
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.repeat.set(tileRep, tileRep);
      t.needsUpdate = true;
    });
  }, [tileRep]);
  useEffect(() => {
    const { mat } = threeR.current;
    if (!mat || !srcURL) return;
    new THREE.TextureLoader().load(srcURL, (t) => {
      t.colorSpace = THREE.SRGBColorSpace;
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.repeat.set(tileRep, tileRep);
      mat.map?.dispose();
      mat.map = t;
      mat.needsUpdate = true;
    });
  }, [srcURL]);
  useEffect(() => {
    const { mat } = threeR.current;
    if (!mat || !maps) return;
    const L = new THREE.TextureLoader();
    const ld = (u, cb) =>
      L.load(u, (t) => {
        t.wrapS = t.wrapT = THREE.RepeatWrapping;
        t.repeat.set(tileRep, tileRep);
        cb(t);
      });
    ld(maps.normal, (t) => {
      mat.normalMap?.dispose();
      mat.normalMap = t;
      mat.normalScale.set(1, 1);
      mat.needsUpdate = true;
    });
    ld(maps.roughness, (t) => {
      mat.roughnessMap?.dispose();
      mat.roughnessMap = t;
      mat.roughness = 1;
      mat.needsUpdate = true;
    });
    ld(maps.metallic, (t) => {
      mat.metalnessMap?.dispose();
      mat.metalnessMap = t;
      mat.metalness = 1;
      mat.needsUpdate = true;
    });
    ld(maps.ao, (t) => {
      mat.aoMap?.dispose();
      mat.aoMap = t;
      mat.aoMapIntensity = 1;
      mat.needsUpdate = true;
    });
  }, [maps]);

  const ss = (k) => (e) =>
    setSettings((s) => ({ ...s, [k]: parseFloat(e.target.value) }));
  const tg = (k) => () => setSettings((s) => ({ ...s, [k]: !s[k] }));
  const dl = (k) => {
    if (!maps?.[k]) return;
    const a = document.createElement("a");
    a.href = maps[k];
    a.download = `unarrived_${k}.png`;
    a.click();
  };
  const dlAll = async () => {
    if (!maps || exporting) return;
    setExporting(true);
    try {
      await doZip(maps, res);
    } finally {
      setExporting(false);
    }
  };
  const analyze = () => {
    if (!imgR.current || proc) return;
    setAiStep("analyzing");
    setTimeout(() => {
      const h = analyzeAlbedo(imgR.current);
      setHint(h);
      setSettings((s) => ({ ...s, ...h }));
      setAiStep("ready");
    }, 80);
  };

  const step = STEPS[aiStep];
  const availRes = ALL_RES.filter((r) => r <= gpu.maxTex);

  return (
    <>
      <style>{`
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
      *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
      html,body{height:100%;overflow:hidden;background:${C.bg}}
      body{font-family:'Inter',sans-serif;font-size:13px;color:${C.text};-webkit-font-smoothing:antialiased}
      ::-webkit-scrollbar{width:4px}
      ::-webkit-scrollbar-track{background:${C.sidebar}}
      ::-webkit-scrollbar-thumb{background:${C.border};border-radius:4px}
      ::-webkit-scrollbar-thumb:hover{background:${C.borderHi}}
      input[type=range]{-webkit-appearance:none;width:100%;height:3px;background:${C.panel};outline:none;cursor:pointer;border-radius:2px}
      input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:${C.accent};cursor:pointer;border:2px solid ${C.bg};box-shadow:0 0 0 1px ${C.accentDim}}
      input[type=range]:hover{background:${C.border}}
      @keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
      @keyframes spin{to{transform:rotate(360deg)}}
      @keyframes fadein{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
      button{font-family:'Inter',sans-serif;transition:all .15s}
      button:hover:not(:disabled){filter:brightness(1.15)}
    `}</style>

      <div
        style={{
          display: "flex",
          height: "100vh",
          background: C.bg,
          overflow: "hidden",
        }}
      >
        {/* ═══ LEFT SIDEBAR ═══ */}
        <aside
          style={{
            width: 280,
            flexShrink: 0,
            borderRight: `1px solid ${C.border}`,
            display: "flex",
            flexDirection: "column",
            background: C.sidebar,
            overflowY: "auto",
          }}
        >
          {/* Header */}
          <div
            style={{
              padding: "20px 20px 16px",
              borderBottom: `1px solid ${C.border}`,
            }}
          >
            <div
              style={{
                fontSize: 10,
                letterSpacing: 4,
                color: C.textFaint,
                fontFamily: "'JetBrains Mono',monospace",
                marginBottom: 4,
              }}
            >
              UNARRIVED
            </div>
            <div
              style={{
                fontSize: 24,
                fontWeight: 600,
                color: C.accent,
                letterSpacing: 1,
                lineHeight: 1,
              }}
            >
              MAPPER
            </div>
            <div
              style={{
                fontSize: 11,
                color: C.textDim,
                marginTop: 6,
                fontFamily: "'JetBrains Mono',monospace",
              }}
            >
              Smart PBR · Material AI · v6
            </div>
          </div>

          {/* Drop zone */}
          <div style={{ padding: "14px 14px 8px" }}>
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                loadFile(e.dataTransfer.files[0]);
              }}
              onClick={() => {
                const i = document.createElement("input");
                i.type = "file";
                i.accept = "image/*";
                i.onchange = (e) => loadFile(e.target.files[0]);
                i.click();
              }}
              style={{
                border: `2px dashed ${dragOver ? C.accent : C.border}`,
                borderRadius: 8,
                cursor: "pointer",
                background: dragOver ? "#1c1400" : C.panel,
                transition: "all .2s",
                overflow: "hidden",
              }}
            >
              {srcURL ? (
                <img
                  src={srcURL}
                  style={{
                    width: "100%",
                    height: 100,
                    objectFit: "cover",
                    display: "block",
                  }}
                />
              ) : (
                <div style={{ padding: "28px 16px", textAlign: "center" }}>
                  <div style={{ fontSize: 28, marginBottom: 8 }}>⬡</div>
                  <div
                    style={{ fontSize: 13, color: C.textDim, fontWeight: 500 }}
                  >
                    Drop albedo image here
                  </div>
                  <div
                    style={{ fontSize: 11, color: C.textFaint, marginTop: 4 }}
                  >
                    or click to browse
                  </div>
                </div>
              )}
              {srcURL && (
                <div
                  style={{
                    padding: "6px 12px",
                    fontSize: 11,
                    color: C.textDim,
                    background: C.sidebar,
                    borderTop: `1px solid ${C.border}`,
                  }}
                >
                  {imgInfo && `${imgInfo.w} × ${imgInfo.h}px · `}Click to
                  replace
                </div>
              )}
            </div>
          </div>

          {/* Mode toggle */}
          <div
            style={{
              padding: "10px 14px 8px",
              borderBottom: `1px solid ${C.border}`,
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "8px 12px",
                background: C.panel,
                borderRadius: 6,
                border: `1px solid ${quality ? C.purple + "66" : C.border}`,
              }}
            >
              <div>
                <div
                  style={{
                    fontSize: 12,
                    fontWeight: 600,
                    color: quality ? C.purple : C.text,
                  }}
                >
                  {quality ? "Quality Mode" : "Fast Mode"}
                </div>
                <div style={{ fontSize: 11, color: C.textDim, marginTop: 1 }}>
                  {quality
                    ? "+ Material Segmentation"
                    : "Depth-Anything V2 only"}
                </div>
              </div>
              <Toggle
                value={quality}
                onChange={() => setQuality((v) => !v)}
                color={C.purple}
              />
            </div>
          </div>

          {/* Status bar */}
          <div
            style={{
              padding: "10px 14px",
              borderBottom: `1px solid ${C.border}`,
              display: "flex",
              flexDirection: "column",
              gap: 6,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: step.dot,
                  flexShrink: 0,
                  animation: proc ? "blink 1s infinite" : "none",
                }}
              />
              <div
                style={{
                  fontSize: 12,
                  color: proc ? step.dot : C.textDim,
                  flex: 1,
                  fontWeight: proc ? 500 : 400,
                }}
              >
                {step.label}
              </div>
              <button
                onClick={analyze}
                disabled={!imgR.current || proc}
                style={{
                  padding: "3px 10px",
                  borderRadius: 4,
                  fontSize: 11,
                  fontWeight: 500,
                  border: `1px solid ${imgR.current && !proc ? C.pink : C.border}`,
                  background: "transparent",
                  color: imgR.current && !proc ? C.pink : C.textFaint,
                  cursor: imgR.current && !proc ? "pointer" : "not-allowed",
                }}
              >
                Analyze
              </button>
            </div>
            {tileProg && (
              <div>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: 11,
                    color: C.textDim,
                    marginBottom: 4,
                  }}
                >
                  <span>
                    Processing tile {tileProg.c} of {tileProg.t}
                  </span>
                  <span>{Math.round((tileProg.c / tileProg.t) * 100)}%</span>
                </div>
                <div
                  style={{ height: 3, background: C.panel, borderRadius: 2 }}
                >
                  <div
                    style={{
                      height: "100%",
                      background: C.accent,
                      borderRadius: 2,
                      width: `${(tileProg.c / tileProg.t) * 100}%`,
                      transition: "width .2s",
                    }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Analyze hint */}
          {hint && (
            <div
              style={{
                padding: "10px 14px",
                background: "#140d1f",
                borderBottom: `1px solid ${C.purple}33`,
              }}
            >
              <div
                style={{
                  fontSize: 11,
                  color: C.purple,
                  fontWeight: 600,
                  marginBottom: 6,
                }}
              >
                ✦ AI Analysis Applied
              </div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "3px 12px",
                }}
              >
                {Object.entries(hint).map(([k, v]) => (
                  <div
                    key={k}
                    style={{
                      fontSize: 10,
                      color: C.textDim,
                      fontFamily: "'JetBrains Mono',monospace",
                    }}
                  >
                    <span style={{ color: C.textFaint }}>
                      {k.replace(/([A-Z])/g, " $1").toLowerCase()}
                    </span>
                    <span style={{ color: C.purple, marginLeft: 4 }}>
                      {typeof v === "number" ? v.toFixed(2) : String(v)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Controls */}
          <div style={{ padding: "8px 14px 14px", flex: 1 }}>
            <Section label="Normal Map">
              <Slider
                label="Depth Strength"
                v={settings.normalStr}
                min={0.5}
                max={14}
                step={0.1}
                onChange={ss("normalStr")}
              />
              <Slider
                label="Detail Mix"
                v={settings.detailMix}
                min={0}
                max={1}
                step={0.01}
                onChange={ss("detailMix")}
                hint="Highpass albedo → normal contribution"
              />
              <Slider
                label="Bilateral σ"
                v={settings.normalSigma}
                min={0.01}
                max={0.3}
                step={0.01}
                onChange={ss("normalSigma")}
                hint="Depth edge preservation"
              />
            </Section>

            <Section label="Displacement">
              <Slider
                label="Midpoint"
                v={settings.dispMid}
                min={0}
                max={1}
                step={0.01}
                onChange={ss("dispMid")}
                hint="Neutral depth level"
              />
              <Slider
                label="Contrast"
                v={settings.dispCon}
                min={0.5}
                max={6}
                step={0.1}
                onChange={ss("dispCon")}
              />
              <Toggle
                label="Invert"
                value={settings.dispInv}
                onChange={tg("dispInv")}
              />
            </Section>

            <Section label="Roughness">
              <Slider
                label="Contrast"
                v={settings.roughCon}
                min={0.5}
                max={6}
                step={0.1}
                onChange={ss("roughCon")}
              />
              <Slider
                label="Bias"
                v={settings.roughBias}
                min={-0.5}
                max={0.5}
                step={0.01}
                onChange={ss("roughBias")}
              />
              <Slider
                label="Freq Mix"
                v={settings.roughDetailMix}
                min={0}
                max={1}
                step={0.01}
                onChange={ss("roughDetailMix")}
                hint="0 = depth lum · 1 = surface complexity"
              />
              <Slider
                label="Material Blend"
                v={settings.matBlend}
                min={0}
                max={1}
                step={0.01}
                onChange={ss("matBlend")}
                hint="0 = heuristic · 1 = class preset"
              />
              <Toggle
                label="Invert"
                value={settings.roughInv}
                onChange={tg("roughInv")}
              />
            </Section>

            <Section label="Metallic">
              <Slider
                label="Threshold"
                v={settings.metalThr}
                min={0}
                max={1}
                step={0.01}
                onChange={ss("metalThr")}
              />
              <Slider
                label="Softness"
                v={settings.metalSoft}
                min={0.01}
                max={0.3}
                step={0.01}
                onChange={ss("metalSoft")}
              />
            </Section>

            <Section label="Ambient Occlusion (HBAO)">
              <Slider
                label="Radius"
                v={settings.aoRadius}
                min={0.1}
                max={3}
                step={0.05}
                onChange={ss("aoRadius")}
              />
              <Slider
                label="Strength"
                v={settings.aoStr}
                min={0}
                max={3}
                step={0.05}
                onChange={ss("aoStr")}
              />
            </Section>

            <Section label="Curvature">
              <Slider
                label="Scale"
                v={settings.curvScale}
                min={0.1}
                max={4}
                step={0.05}
                onChange={ss("curvScale")}
                hint="Multi-scale: 1px / 4px / 12px blend"
              />
            </Section>

            <Section label="Frequency Separation">
              <Slider
                label="Blur Radius"
                v={settings.blurRadius}
                min={1}
                max={24}
                step={0.5}
                onChange={ss("blurRadius")}
                hint="Larger = wider detail extraction"
              />
            </Section>
          </div>

          {/* GPU + Resolution */}
          <div
            style={{ padding: "12px 14px", borderTop: `1px solid ${C.border}` }}
          >
            <div
              style={{
                fontSize: 11,
                color: C.textFaint,
                fontFamily: "'JetBrains Mono',monospace",
                marginBottom: 10,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {gpu.renderer} · max {gpu.maxTex}px
            </div>
            <Label>Output Resolution</Label>
            <div style={{ display: "flex", gap: 4, marginTop: 6 }}>
              {ALL_RES.map((r) => {
                const avail = r <= gpu.maxTex,
                  active = r === res,
                  warn = r >= 4096;
                return (
                  <button
                    key={r}
                    onClick={() => avail && setRes(r)}
                    disabled={!avail}
                    style={{
                      flex: 1,
                      padding: "6px 4px",
                      borderRadius: 4,
                      fontSize: 11,
                      fontWeight: active ? 600 : 400,
                      cursor: avail ? "pointer" : "not-allowed",
                      background: active ? C.accent + 22 : "transparent",
                      border: `1px solid ${active ? C.accent : avail ? C.border : C.border + "44"}`,
                      color: active
                        ? C.accent
                        : avail
                          ? warn
                            ? C.yellow
                            : C.text
                          : C.textFaint,
                    }}
                  >
                    {r >= 1000 ? `${r / 1000}K` : r}
                  </button>
                );
              })}
            </div>
            {res >= 4096 && (
              <div
                style={{
                  marginTop: 8,
                  padding: "6px 10px",
                  background: "#1c1400",
                  border: `1px solid ${C.accent}33`,
                  borderRadius: 4,
                  fontSize: 11,
                  color: C.accentDim,
                }}
              >
                ⚠ Tiled rendering · ~{vramMB(TILE)}MB VRAM/tile
              </div>
            )}
          </div>

          {/* Export */}
          <div
            style={{
              padding: "12px 14px 16px",
              borderTop: `1px solid ${C.border}`,
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 10,
              }}
            >
              <Label>Export</Label>
              <button
                onClick={dlAll}
                disabled={!maps || exporting}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  padding: "6px 12px",
                  borderRadius: 4,
                  fontSize: 12,
                  fontWeight: 500,
                  background:
                    maps && !exporting ? C.blue + "22" : "transparent",
                  border: `1px solid ${maps && !exporting ? C.blue : C.border}`,
                  color: maps && !exporting ? C.blue : C.textFaint,
                  cursor: maps && !exporting ? "pointer" : "not-allowed",
                }}
              >
                {exporting ? (
                  <span
                    style={{
                      animation: "spin 1s linear infinite",
                      display: "inline-block",
                      fontSize: 14,
                    }}
                  >
                    ⟳
                  </span>
                ) : (
                  <span>⬡</span>
                )}
                {exporting ? "Packing…" : "Export All ZIP"}
              </button>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
              {ALL_MAPS.map((m) => (
                <button
                  key={m.key}
                  onClick={() => dl(m.key)}
                  disabled={!maps?.[m.key]}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "7px 10px",
                    borderRadius: 4,
                    background: maps?.[m.key] ? "transparent" : C.panel + "88",
                    border: `1px solid ${maps?.[m.key] ? C.border : C.border + "66"}`,
                    color: maps?.[m.key] ? C.text : C.textFaint,
                    cursor: maps?.[m.key] ? "pointer" : "not-allowed",
                    textAlign: "left",
                  }}
                >
                  <div
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: 2,
                      background: maps?.[m.key] ? m.color : C.textFaint,
                      flexShrink: 0,
                    }}
                  />
                  <span style={{ fontSize: 12, flex: 1 }}>{m.label}</span>
                  {m.key === "orm" && (
                    <span
                      style={{
                        fontSize: 10,
                        color: C.textDim,
                        fontFamily: "'JetBrains Mono',monospace",
                      }}
                    >
                      UE/Unity
                    </span>
                  )}
                  {maps?.[m.key] && (
                    <span style={{ fontSize: 12, color: C.textDim }}>↓</span>
                  )}
                </button>
              ))}
            </div>
          </div>
        </aside>

        {/* ═══ CENTER: MAP GRID ═══ */}
        <main
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            minWidth: 0,
            background: C.bg,
          }}
        >
          {/* Top bar */}
          <div
            style={{
              height: 44,
              borderBottom: `1px solid ${C.border}`,
              display: "flex",
              alignItems: "center",
              padding: "0 20px",
              gap: 20,
              flexShrink: 0,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: "50%",
                  background: proc ? step.dot : maps ? C.green : C.textFaint,
                  animation: proc ? "blink 1s infinite" : "none",
                }}
              />
              <span style={{ fontSize: 12, color: C.textDim }}>
                {tileProg ? `Tile ${tileProg.c}/${tileProg.t}` : step.label}
              </span>
            </div>
            {imgInfo && (
              <span style={{ fontSize: 12, color: C.textFaint }}>
                Source: {imgInfo.w}×{imgInfo.h}px
              </span>
            )}
            <span style={{ fontSize: 12, color: C.textFaint }}>
              Output: {res}px
            </span>
            <div
              style={{
                marginLeft: "auto",
                fontSize: 11,
                color: C.textFaint,
                fontFamily: "'JetBrains Mono',monospace",
              }}
            >
              {quality ? "depth-anything-v2 + segformer" : "depth-anything-v2"}{" "}
              · Bilateral · HBAO · ORM
            </div>
          </div>

          {/* 3×2 grid */}
          <div
            style={{
              flex: 1,
              display: "grid",
              gridTemplateColumns: "1fr 1fr 1fr",
              gridTemplateRows: "1fr 1fr",
              gap: 2,
              padding: 2,
              background: C.bg,
              overflow: "hidden",
            }}
          >
            {MAPS_DEF.map((m) => (
              <MapTile
                key={m.key}
                meta={m}
                url={maps?.[m.key]}
                hovered={hovered === m.key}
                onEnter={() => setHovered(m.key)}
                onLeave={() => setHovered(null)}
                onDl={() => dl(m.key)}
              />
            ))}
          </div>
        </main>

        {/* ═══ RIGHT: 3D PREVIEW ═══ */}
        <aside
          style={{
            width: 280,
            flexShrink: 0,
            borderLeft: `1px solid ${C.border}`,
            display: "flex",
            flexDirection: "column",
            background: C.sidebar,
          }}
        >
          <div
            style={{
              height: 44,
              borderBottom: `1px solid ${C.border}`,
              display: "flex",
              alignItems: "center",
              padding: "0 16px",
            }}
          >
            <span style={{ fontSize: 13, fontWeight: 600, color: C.text }}>
              PBR Preview
            </span>
            <span
              style={{ marginLeft: "auto", fontSize: 11, color: C.textFaint }}
            >
              Three.js r128
            </span>
          </div>

          <div style={{ flex: 1, position: "relative" }}>
            <canvas
              ref={cvs}
              style={{
                width: "100%",
                height: "100%",
                display: "block",
                cursor: "grab",
              }}
            />
            {!srcURL && (
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  pointerEvents: "none",
                }}
              >
                <div style={{ textAlign: "center", color: C.textFaint }}>
                  <div style={{ fontSize: 28, marginBottom: 8 }}>⬡</div>
                  <div style={{ fontSize: 13 }}>Load image to preview</div>
                </div>
              </div>
            )}
          </div>

          {/* Shape */}
          <div
            style={{ padding: "10px 12px", borderTop: `1px solid ${C.border}` }}
          >
            <Label>Preview Shape</Label>
            <div style={{ display: "flex", gap: 4, marginTop: 6 }}>
              {SHAPES.map((s) => (
                <button
                  key={s.k}
                  onClick={() => setShape(s.k)}
                  style={{
                    flex: 1,
                    padding: "7px 0",
                    borderRadius: 4,
                    fontSize: 12,
                    fontWeight: shape === s.k ? 600 : 400,
                    cursor: "pointer",
                    background: shape === s.k ? C.accent + "22" : "transparent",
                    border: `1px solid ${shape === s.k ? C.accent : C.border}`,
                    color: shape === s.k ? C.accent : C.textDim,
                  }}
                >
                  {s.l}
                </button>
              ))}
            </div>
          </div>

          {/* Tile repeat */}
          <div
            style={{ padding: "10px 12px", borderTop: `1px solid ${C.border}` }}
          >
            <Label>Tile Repeat</Label>
            <div style={{ display: "flex", gap: 4, marginTop: 6 }}>
              {[1, 2, 4].map((n) => (
                <button
                  key={n}
                  onClick={() => setTileRep(n)}
                  style={{
                    flex: 1,
                    padding: "7px 0",
                    borderRadius: 4,
                    fontSize: 12,
                    fontWeight: tileRep === n ? 600 : 400,
                    cursor: "pointer",
                    background: tileRep === n ? C.blue + "22" : "transparent",
                    border: `1px solid ${tileRep === n ? C.blue : C.border}`,
                    color: tileRep === n ? C.blue : C.textDim,
                  }}
                >
                  {n}×
                </button>
              ))}
            </div>
          </div>

          {/* Environment */}
          <div
            style={{ padding: "10px 12px", borderTop: `1px solid ${C.border}` }}
          >
            <Label>Environment (HDRI)</Label>
            <div style={{ display: "flex", gap: 4, marginTop: 6 }}>
              {Object.keys(HDRI).map((k) => (
                <button
                  key={k}
                  onClick={() => setHdri(k)}
                  style={{
                    flex: 1,
                    padding: "7px 0",
                    borderRadius: 4,
                    fontSize: 11,
                    fontWeight: hdri === k ? 600 : 400,
                    cursor: "pointer",
                    background: hdri === k ? C.teal + "22" : "transparent",
                    border: `1px solid ${hdri === k ? C.teal : C.border}`,
                    color: hdri === k ? C.teal : C.textDim,
                  }}
                >
                  {k.charAt(0).toUpperCase() + k.slice(1)}
                </button>
              ))}
            </div>
          </div>

          <div
            style={{
              padding: "10px 12px",
              borderTop: `1px solid ${C.border}`,
              fontSize: 11,
              color: C.textFaint,
            }}
          >
            Drag to orbit · Auto-rotates when idle
          </div>

          {/* Map legend */}
          <div
            style={{
              padding: "10px 12px 14px",
              borderTop: `1px solid ${C.border}`,
            }}
          >
            <Label>Channels</Label>
            <div
              style={{
                marginTop: 8,
                display: "flex",
                flexDirection: "column",
                gap: 5,
              }}
            >
              {ALL_MAPS.map((m) => (
                <div
                  key={m.key}
                  style={{ display: "flex", alignItems: "center", gap: 8 }}
                >
                  <div
                    style={{
                      width: 10,
                      height: 10,
                      borderRadius: 2,
                      background: m.color,
                      flexShrink: 0,
                    }}
                  />
                  <span style={{ fontSize: 11, color: C.textDim, flex: 1 }}>
                    {m.label}
                  </span>
                  <span
                    style={{
                      fontSize: 10,
                      color: C.textFaint,
                      fontFamily: "'JetBrains Mono',monospace",
                    }}
                  >
                    {m.hint.split("·")[0].trim()}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Pipeline info */}
          <div
            style={{
              padding: "10px 12px 14px",
              borderTop: `1px solid ${C.border}`,
            }}
          >
            <Label>Pipeline</Label>
            <div
              style={{
                marginTop: 8,
                display: "flex",
                flexDirection: "column",
                gap: 3,
              }}
            >
              {[
                ["Depth", "depth-anything-v2"],
                ["Seg", "segformer-b0-ade (quality)"],
                ["Normal", "Bilateral Sobel GLSL3"],
                ["AO", "HBAO 32-sample spiral"],
                ["Curv", "Multi-scale 1/4/12px"],
                ["8K", "Tiled 2048×N + 32px pad"],
              ].map(([k, v]) => (
                <div
                  key={k}
                  style={{
                    display: "flex",
                    gap: 8,
                    fontSize: 10,
                    fontFamily: "'JetBrains Mono',monospace",
                  }}
                >
                  <span style={{ color: C.textFaint, minWidth: 40 }}>{k}</span>
                  <span style={{ color: C.textDim }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </>
  );
}

/* ═══════════════════════════════════════════════════════════════
   SUB-COMPONENTS
═══════════════════════════════════════════════════════════════ */
function Label({ children }) {
  return (
    <div
      style={{
        fontSize: 11,
        fontWeight: 600,
        color: C.textDim,
        letterSpacing: 0.5,
        textTransform: "uppercase",
      }}
    >
      {children}
    </div>
  );
}
function Section({ label, children }) {
  return (
    <div
      style={{
        paddingTop: 16,
        marginTop: 4,
        borderTop: `1px solid ${C.border}`,
      }}
    >
      <Label>{label}</Label>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 12,
          marginTop: 10,
        }}
      >
        {children}
      </div>
    </div>
  );
}
function Slider({ label, v, min, max, step, onChange, hint }) {
  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 6,
        }}
      >
        <span style={{ fontSize: 12, color: C.text }}>{label}</span>
        <span
          style={{
            fontSize: 12,
            fontFamily: "'JetBrains Mono',monospace",
            fontWeight: 500,
            color: C.accent,
          }}
        >
          {v.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={v}
        onChange={onChange}
      />
      {hint && (
        <div style={{ fontSize: 10, color: C.textFaint, marginTop: 4 }}>
          {hint}
        </div>
      )}
    </div>
  );
}
function Toggle({ label, value, onChange, color = C.accent }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "2px 0",
      }}
    >
      {label && <span style={{ fontSize: 12, color: C.text }}>{label}</span>}
      <div
        onClick={onChange}
        style={{
          width: 34,
          height: 18,
          borderRadius: 9,
          cursor: "pointer",
          position: "relative",
          background: value ? color : C.panel,
          transition: "background .2s",
          border: `1px solid ${value ? color : C.border}`,
          flexShrink: 0,
        }}
      >
        <div
          style={{
            position: "absolute",
            top: 2,
            width: 12,
            height: 12,
            borderRadius: 6,
            background: value ? "#fff" : C.textDim,
            left: value ? 18 : 2,
            transition: "left .2s",
          }}
        />
      </div>
    </div>
  );
}
function MapTile({ meta, url, hovered, onEnter, onLeave, onDl }) {
  return (
    <div
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
      style={{
        background: "#09090f",
        position: "relative",
        overflow: "hidden",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        borderRadius: 4,
      }}
    >
      {url ? (
        <img
          src={url}
          alt={meta.label}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "contain",
            display: "block",
            animation: "fadein .3s ease",
          }}
        />
      ) : (
        <div style={{ textAlign: "center", padding: 20 }}>
          <div style={{ fontSize: 24, marginBottom: 8, opacity: 0.2 }}>◫</div>
          <div style={{ fontSize: 12, color: C.textFaint }}>{meta.label}</div>
          <div
            style={{ fontSize: 10, color: C.textFaint + "88", marginTop: 4 }}
          >
            Awaiting AI depth
          </div>
        </div>
      )}

      {/* Top label */}
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          padding: "8px 10px",
          background: "linear-gradient(to bottom,rgba(9,9,15,.9),transparent)",
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <div
          style={{
            width: 8,
            height: 8,
            borderRadius: 2,
            background: meta.color,
            flexShrink: 0,
          }}
        />
        <span
          style={{
            fontSize: 12,
            fontWeight: 600,
            color: "#fff",
            textShadow: "0 1px 3px rgba(0,0,0,.8)",
          }}
        >
          {meta.label}
        </span>
        {url && hovered && (
          <button
            onClick={onDl}
            style={{
              marginLeft: "auto",
              padding: "2px 8px",
              borderRadius: 3,
              fontSize: 11,
              background: "rgba(255,255,255,.15)",
              border: "1px solid rgba(255,255,255,.3)",
              color: "#fff",
              cursor: "pointer",
              backdropFilter: "blur(4px)",
            }}
          >
            ↓ Save
          </button>
        )}
      </div>

      {/* Bottom hint */}
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          padding: "8px 10px",
          background: "linear-gradient(to top,rgba(9,9,15,.8),transparent)",
          fontSize: 10,
          color: "rgba(255,255,255,.4)",
          textShadow: "0 1px 2px rgba(0,0,0,.8)",
        }}
      >
        {meta.hint}
      </div>
    </div>
  );
}
