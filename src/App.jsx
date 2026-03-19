import { useState, useRef, useEffect, useCallback } from "react";
import * as THREE from "three";
import { pipeline } from "@huggingface/transformers";
import JSZip from "jszip";

/* ═══════════════════════════════════════════════════════════════
   AI SINGLETONS
   depth-anything-v2: MiDaS 대비 정확도 대폭 향상
   segformer: ADE20k 150-class 재질 분류 → material mask
═══════════════════════════════════════════════════════════════ */
let _depthPromise = null;
let _segPromise = null;

const getDepth = () => {
  if (!_depthPromise)
    _depthPromise = pipeline(
      "depth-estimation",
      "Xenova/depth-anything-v2",
    ).catch((e) => {
      _depthPromise = null;
      throw e;
    });
  return _depthPromise;
};

const getSegmenter = () => {
  if (!_segPromise)
    _segPromise = pipeline(
      "image-segmentation",
      "Xenova/segformer-b0-finetuned-ade-512-512",
    ).catch((e) => {
      _segPromise = null;
      throw e;
    });
  return _segPromise;
};

/* ═══════════════════════════════════════════════════════════════
   GPU CAPABILITY QUERY
═══════════════════════════════════════════════════════════════ */
function queryGPU() {
  try {
    const c = document.createElement("canvas");
    const gl = c.getContext("webgl2");
    if (!gl) return { maxTex: 2048, renderer: "WebGL2 unavailable" };
    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const renderer = ext
      ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)
      : "GPU info unavailable";
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return { maxTex, renderer };
  } catch {
    return { maxTex: 2048, renderer: "Query failed" };
  }
}

// 파이프라인 중간 텍스처 수 기반 VRAM 추정 (MB)
const vramMB = (res) => Math.round((res * res * 4 * 12) / 1024 / 1024);

/* ═══════════════════════════════════════════════════════════════
   PROCEDURAL HDRI
═══════════════════════════════════════════════════════════════ */
const HDRI_PRESETS = {
  studio: {
    label: "STUDIO",
    sky: ["#0d1020", "#1a2040", "#404860", "#707890"],
    key: { color: "#ffe8c8", x: 140, y: 55, r: 90 },
    fill: { color: "#c8d8ff", x: 380, y: 100, r: 60 },
    ground: "#181020",
  },
  outdoor: {
    label: "OUTDOOR",
    sky: ["#0a1a40", "#1a4090", "#4090d0", "#90c8f0"],
    key: { color: "#fff4d0", x: 100, y: 40, r: 110 },
    fill: { color: "#80b0e0", x: 450, y: 80, r: 80 },
    ground: "#2a3018",
  },
  warm: {
    label: "WARM",
    sky: ["#100810", "#302030", "#704830", "#c07040"],
    key: { color: "#ffd090", x: 160, y: 50, r: 100 },
    fill: { color: "#ff8040", x: 400, y: 90, r: 55 },
    ground: "#201008",
  },
};
function buildEnvMap(renderer, key) {
  const p = HDRI_PRESETS[key];
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
    g.addColorStop(0.4, color + "66");
    g.addColorStop(1, color + "00");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
  });
  const tex = new THREE.CanvasTexture(c);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  const pmrem = new THREE.PMREMGenerator(renderer);
  pmrem.compileEquirectangularShader();
  const env = pmrem.fromEquirectangular(tex).texture;
  pmrem.dispose();
  tex.dispose();
  return env;
}

/* ═══════════════════════════════════════════════════════════════
   MATERIAL CLASS MAPPING  (ADE20k label → 0..4)
   0 = organic/dielectric  metal≈0  rough≈0.78
   1 = metal               metal≈0.9 rough≈0.38
   2 = stone/concrete      metal≈0  rough≈0.86
   3 = fabric/soft         metal≈0  rough≈0.92
   4 = plastic/glass       metal≈0  rough≈0.22
═══════════════════════════════════════════════════════════════ */
const LABEL_TO_CLASS = (() => {
  const m = {};
  const assign = (cls, labels) =>
    labels.forEach((l) => {
      m[l] = cls;
    });
  assign(1, [
    "car",
    "bus",
    "truck",
    "van",
    "train",
    "airplane",
    "boat",
    "bicycle",
    "motorbike",
    "radiator",
    "refrigerator",
    "sink",
    "oven",
    "microwave",
    "washing machine",
    "dishwasher",
  ]);
  assign(2, [
    "wall",
    "floor",
    "ceiling",
    "building",
    "column",
    "bridge",
    "pavement",
    "path",
    "rock",
    "stone",
    "stairway",
    "steps",
  ]);
  assign(3, [
    "curtain",
    "sofa",
    "bed",
    "pillow",
    "blanket",
    "mat",
    "carpet",
    "rug",
    "cushion",
  ]);
  assign(4, [
    "bottle",
    "glass",
    "window",
    "jar",
    "cup",
    "bowl",
    "vase",
    "lamp",
  ]);
  // default 0: tree, table, chair, grass, plant, wood, etc.
  return m;
})();

function labelToClass(label) {
  const l = label.toLowerCase();
  for (const [k, v] of Object.entries(LABEL_TO_CLASS)) {
    if (l.includes(k)) return v;
  }
  return 0;
}

// Segmentation → single-channel material mask canvas (512×512)
async function buildMaterialMask(segments) {
  const S = 512;
  const c = document.createElement("canvas");
  c.width = c.height = S;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  const id = ctx.createImageData(S, S);

  // Default class 0 (encoded as 0)
  id.data.fill(0);
  for (let i = 3; i < id.data.length; i += 4) id.data[i] = 255;

  for (const seg of segments || []) {
    if (!seg?.mask) continue;
    const cls = labelToClass(seg.label || "");
    const encoded = Math.round((cls / 4) * 255);
    try {
      const mc = seg.mask.toCanvas ? seg.mask.toCanvas() : null;
      if (!mc) continue;
      const tmpC = document.createElement("canvas");
      tmpC.width = tmpC.height = S;
      tmpC.getContext("2d").drawImage(mc, 0, 0, S, S);
      const md = tmpC.getContext("2d").getImageData(0, 0, S, S).data;
      for (let i = 0; i < id.data.length; i += 4) {
        if (md[i] > 128) {
          id.data[i] = encoded;
          id.data[i + 1] = 0;
          id.data[i + 2] = 0;
        }
      }
    } catch {
      /* skip bad masks */
    }
  }

  ctx.putImageData(id, 0, 0);
  return c;
}

/* ═══════════════════════════════════════════════════════════════
   GLSL3 SHADERS  — v6 (Bilateral / HBAO / MultiCurv / FreqRough)
═══════════════════════════════════════════════════════════════ */
const VS = `#version 300 es
in vec2 a; out vec2 v;
void main(){ v=a*.5+.5; gl_Position=vec4(a,0.,1.); }`;

const FS_PASS = `#version 300 es precision highp float;
uniform sampler2D T; in vec2 v; out vec4 o;
void main(){ o=texture(T,v); }`;

const FS_BLUR_H = `#version 300 es precision highp float;
uniform sampler2D T; uniform vec2 px; uniform float radius;
in vec2 v; out vec4 o;
const float W[5]=float[](0.227027,0.194595,0.121622,0.054054,0.016216);
void main(){ vec4 c=texture(T,v)*W[0];
  for(int i=1;i<5;i++) c+=texture(T,v+vec2(float(i)*px.x*radius,0.))*W[i]+texture(T,v-vec2(float(i)*px.x*radius,0.))*W[i];
  o=c; }`;

const FS_BLUR_V = `#version 300 es precision highp float;
uniform sampler2D T; uniform vec2 px; uniform float radius;
in vec2 v; out vec4 o;
const float W[5]=float[](0.227027,0.194595,0.121622,0.054054,0.016216);
void main(){ vec4 c=texture(T,v)*W[0];
  for(int i=1;i<5;i++) c+=texture(T,v+vec2(0.,float(i)*px.y*radius))*W[i]+texture(T,v-vec2(0.,float(i)*px.y*radius))*W[i];
  o=c; }`;

const FS_HIGHPASS = `#version 300 es precision highp float;
uniform sampler2D ORIG,BLUR; in vec2 v; out vec4 o;
void main(){ o=clamp(texture(ORIG,v)-texture(BLUR,v)+vec4(.5),0.,1.); }`;

// ── BILATERAL NORMAL ─────────────────────────────────────────
// 각 Sobel 샘플을 depth 유사도로 가중 → 깊이 경계에서 노멀 번짐 방지
const FS_NORMAL = `#version 300 es precision highp float;
uniform sampler2D DEPTH,DETAIL; uniform vec2 px; uniform float str,detailMix,sigma;
in vec2 v; out vec4 o;
float bilW(float dc,float ds){ float d=dc-ds; return exp(-d*d/(sigma*sigma+0.0001)); }
float hD(vec2 u){ return texture(DEPTH,u).r; }
float hA(vec2 u){ return texture(DETAIL,u).r-.5; }
void main(){
  float dc=hD(v);
  // 3×3 Sobel kernel indices:  0(−1,1) 1(0,1) 2(1,1)  3(−1,0) 4(1,0)  5(−1,−1) 6(0,−1) 7(1,−1)
  vec2 ofs[8]; ofs[0]=px*vec2(-1,1); ofs[1]=px*vec2(0,1); ofs[2]=px*vec2(1,1);
               ofs[3]=px*vec2(-1,0); ofs[4]=px*vec2(1,0);
               ofs[5]=px*vec2(-1,-1); ofs[6]=px*vec2(0,-1); ofs[7]=px*vec2(1,-1);
  float kx[8]; kx[0]=-1.; kx[1]=0.; kx[2]=1.; kx[3]=-2.; kx[4]=2.; kx[5]=-1.; kx[6]=0.; kx[7]=1.;
  float ky[8]; ky[0]=1.;  ky[1]=2.; ky[2]=1.; ky[3]=0.;  ky[4]=0.; ky[5]=-1.; ky[6]=-2.; ky[7]=-1.;
  float dx=0.,dy=0., adx=0.,ady=0.;
  for(int i=0;i<8;i++){
    float ds=hD(v+ofs[i]); float bw=bilW(dc,ds);
    float as=hA(v+ofs[i]);
    dx+=kx[i]*ds*bw; dy+=ky[i]*ds*bw;
    adx+=kx[i]*as;   ady+=ky[i]*as;
  }
  vec3 nD=normalize(vec3(dx*str,dy*str,1.));
  vec3 nA=normalize(vec3(adx*str*1.5,ady*str*1.5,1.));
  vec3 n=normalize(vec3(nD.xy+nA.xy*detailMix,nD.z));
  o=vec4(n*.5+.5,1.);
}`;

// ── DISPLACEMENT ─────────────────────────────────────────────
const FS_DISPLACE = `#version 300 es precision highp float;
uniform sampler2D T; uniform float mid,con,inv;
in vec2 v; out vec4 o;
void main(){ float d=texture(T,v).r; d=clamp((d-mid)*con+0.5,0.,1.); d=mix(d,1.-d,inv); o=vec4(d,d,d,1.); }`;

// ── MATERIAL-AWARE ROUGHNESS  (frequency-energy based) ───────
// localVariance = E[x²]-E[x]² over 5×5 → 미세면 복잡도
// freqEnergy    = highpass magnitude → 고주파 에너지
// materialBias  = segmentation class에 따른 조정
const FS_ROUGH = `#version 300 es precision highp float;
uniform sampler2D DEPTH,DETAIL,SEGMASK; uniform vec2 px;
uniform float con,bias,inv,detailMix,matBlend;
in vec2 v; out vec4 o;
void main(){
  // 5×5 local variance on depth
  float sumD=0.,sumD2=0.;
  for(int dy=-2;dy<=2;dy++) for(int dx=-2;dx<=2;dx++){
    float s=texture(DEPTH,v+px*vec2(float(dx),float(dy))).r;
    sumD+=s; sumD2+=s*s;
  }
  sumD/=25.; sumD2/=25.;
  float localVar=max(sumD2-sumD*sumD,0.)*80.; // amplified variance
  // Highpass frequency energy
  vec3 hi=texture(DETAIL,v).rgb-vec3(.5);
  float freqE=length(hi)*2.;
  // Blend depth lum, local variance, freq energy
  float lBase=texture(DEPTH,v).r;
  float lFreq=mix(lBase, mix(localVar,freqE,.5), detailMix);
  // Material class bias (R channel: class/4 normalized)
  float cls=texture(SEGMASK,v).r*4.;
  float roughPreset=mix(0.75,0.75,0.); // organic default
  roughPreset=mix(roughPreset,0.38,step(0.5,cls)*step(cls,1.5)); // metal
  roughPreset=mix(roughPreset,0.86,step(1.5,cls)*step(cls,2.5)); // stone
  roughPreset=mix(roughPreset,0.92,step(2.5,cls)*step(cls,3.5)); // fabric
  roughPreset=mix(roughPreset,0.22,step(3.5,cls));               // plastic
  float l=mix(lFreq,roughPreset,matBlend);
  l=mix(l,1.-l,inv);
  o=vec4(vec3(clamp((l-.5)*con+.5+bias,0.,1.)),1.);
}`;

// ── MATERIAL-AWARE METALLIC ───────────────────────────────────
const FS_METAL = `#version 300 es precision highp float;
uniform sampler2D T,SEGMASK; uniform float thr,soft;
in vec2 v; out vec4 o;
void main(){
  float l=texture(T,v).r;
  float cls=texture(SEGMASK,v).r*4.;
  // Metal class: threshold 대폭 낮춤 / plastic: 높임
  float bias=0.;
  bias=mix(bias,-0.28, step(0.5,cls)*step(cls,1.5)); // metal: easier
  bias=mix(bias, 0.18, step(3.5,cls));               // plastic: harder
  float m=smoothstep(thr+bias-soft,thr+bias+soft,l);
  o=vec4(m,m,m,1.);
}`;

// ── HBAO (32-sample spiral, depth-range aware) ───────────────
// Fibonacci spiral → 균일 반구 커버리지
// Depth range scaling → 깊이 변화가 큰 영역에서 radius 축소
const FS_AO = `#version 300 es precision highp float;
uniform sampler2D T; uniform vec2 px; uniform float radius,strength;
in vec2 v; out vec4 o;
const float GOLDEN=2.3999632; // golden angle rad
const int SAMPLES=32;
void main(){
  float dc=texture(T,v).r, occ=0.;
  // Local depth range (5×5) → adaptive radius
  float dMin=1.,dMax=0.;
  for(int dy=-2;dy<=2;dy++) for(int dx=-2;dx<=2;dx++){
    float s=texture(T,v+px*vec2(float(dx),float(dy))*4.).r;
    dMin=min(dMin,s); dMax=max(dMax,s);
  }
  float dRange=max(dMax-dMin,0.01);
  float adaptR=radius/dRange; // narrower radius where depth varies a lot
  for(int i=0;i<SAMPLES;i++){
    float r=sqrt(float(i+1)/float(SAMPLES));
    float theta=float(i)*GOLDEN;
    vec2 off=vec2(cos(theta),sin(theta))*px*min(adaptR,radius)*80.*r;
    float ds=texture(T,v+off).r;
    float diff=dc-ds;
    // Cosine-weighted: nearby samples count more
    float cosW=(1.-r)*.7+.3;
    occ+=clamp(diff*8.,0.,1.)*cosW;
  }
  float ao=clamp(1.-(occ/float(SAMPLES)*2.)*strength,0.,1.);
  o=vec4(ao,ao,ao,1.);
}`;

// ── MULTI-SCALE CURVATURE ─────────────────────────────────────
// 3가지 스케일에서 법선 발산 계산 → 블렌드
// Fine(1px): 미세 엣지 / Medium(4px): 중간 디테일 / Coarse(12px): 큰 형태
const FS_CURV = `#version 300 es precision highp float;
uniform sampler2D NORMAL; uniform vec2 px; uniform float scale;
in vec2 v; out vec4 o;
float curvAtScale(float s){
  vec3 n=normalize(texture(NORMAL,v).rgb*2.-1.);
  vec3 nr=normalize(texture(NORMAL,v+vec2(px.x*s,0.)).rgb*2.-1.);
  vec3 nu=normalize(texture(NORMAL,v+vec2(0.,px.y*s)).rgb*2.-1.);
  vec3 nl=normalize(texture(NORMAL,v-vec2(px.x*s,0.)).rgb*2.-1.);
  vec3 nd=normalize(texture(NORMAL,v-vec2(0.,px.y*s)).rgb*2.-1.);
  float div=(dot(n,nr)+dot(n,nu)+dot(n,nl)+dot(n,nd))/4.-1.;
  return clamp(-div*scale*3.+.5,0.,1.);
}
void main(){
  float cf=curvAtScale(1.);
  float cm=curvAtScale(4.);
  float cc=curvAtScale(12.);
  // Fine detail emphasized for edge wear
  float c=cf*.50+cm*.30+cc*.20;
  o=vec4(c,c,c,1.);
}`;

// ── ORM CHANNEL PACK  (Unreal/Unity ready) ───────────────────
// R: AO   G: Roughness   B: Metallic
const FS_ORM = `#version 300 es precision highp float;
uniform sampler2D AO,ROUGH,METAL; in vec2 v; out vec4 o;
void main(){
  float ao=texture(AO,v).r;
  float r=texture(ROUGH,v).r;
  float m=texture(METAL,v).r;
  o=vec4(ao,r,m,1.);
}`;

/* ═══════════════════════════════════════════════════════════════
   TILED PIPELINE
   resolutions ≤ 2048 → single pass
   resolutions > 2048 → TILE(2048) × N tiles, PAD(32) overlap

   핵심:
   - WebGL context를 타일마다 재사용 (VRAM 절약)
   - 타일 사이 PAD 32px: blur/Sobel 커널 경계 artifact 제거
   - 마지막 타일은 실제 이미지 크기만큼만 output에 복사
═══════════════════════════════════════════════════════════════ */
const TILE_SIZE = 2048;
const TILE_PAD = 32;

function buildGLContext(W, H) {
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const gl = c.getContext("webgl2", { preserveDrawingBuffer: true });
  if (!gl) return null;

  const mkProg = (fSrc) => {
    const p = gl.createProgram();
    [
      [gl.VERTEX_SHADER, VS],
      [gl.FRAGMENT_SHADER, fSrc],
    ].forEach(([t, src]) => {
      const sh = gl.createShader(t);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      gl.attachShader(p, sh);
    });
    gl.linkProgram(p);
    return p;
  };

  const P = {
    pass: mkProg(FS_PASS),
    blurH: mkProg(FS_BLUR_H),
    blurV: mkProg(FS_BLUR_V),
    highpass: mkProg(FS_HIGHPASS),
    normal: mkProg(FS_NORMAL),
    displace: mkProg(FS_DISPLACE),
    rough: mkProg(FS_ROUGH),
    metal: mkProg(FS_METAL),
    ao: mkProg(FS_AO),
    curv: mkProg(FS_CURV),
    orm: mkProg(FS_ORM),
  };

  const qBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, qBuf);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
    gl.STATIC_DRAW,
  );

  const mkTex = (src) => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    if (src)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
    else
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
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  };

  const upTex = (t, src) => {
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
  };

  const mkFBO = (t) => {
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      t,
      0,
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return fbo;
  };

  const exec = (prog, fbo, bindings, uniforms) => {
    gl.useProgram(prog);
    const a = gl.getAttribLocation(prog, "a");
    gl.enableVertexAttribArray(a);
    gl.bindBuffer(gl.ARRAY_BUFFER, qBuf);
    gl.vertexAttribPointer(a, 2, gl.FLOAT, false, 0, 0);
    bindings.forEach(([name, tex], i) => {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(gl.getUniformLocation(prog, name), i);
    });
    for (const [k, val] of Object.entries(uniforms)) {
      const u = gl.getUniformLocation(prog, k);
      if (!u) continue;
      Array.isArray(val) ? gl.uniform2fv(u, val) : gl.uniform1f(u, val);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo ?? null);
    gl.viewport(0, 0, W, H);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  };

  // Pre-allocate FBO pool
  const T = {
    alb: mkTex(null),
    dep: mkTex(null),
    seg: mkTex(null),
    bh: mkTex(null),
    bv: mkTex(null),
    hi: mkTex(null),
    norm: mkTex(null),
    ao: mkTex(null),
    rough: mkTex(null),
    metal: mkTex(null),
    displace: mkTex(null),
  };
  const F = {};
  for (const k of Object.keys(T)) F[k] = mkFBO(T[k]);

  return { gl, c, P, T, F, exec, upTex };
}

function toRGBA(src, W, H) {
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  c.getContext("2d", { willReadFrequently: true }).drawImage(src, 0, 0, W, H);
  return c;
}

// Single tile pass → returns output canvas
function runTilePass(ctx, tileAlb, tileDep, tileSeg, W, H, s) {
  const { gl, c, P, T, F, exec, upTex } = ctx;
  const px = [1 / W, 1 / H];

  upTex(T.alb, tileAlb);
  upTex(T.dep, tileDep);
  upTex(T.seg, tileSeg);

  // Frequency separation
  exec(P.blurH, F.bh, [["T", T.alb]], { px, radius: s.blurRadius });
  exec(P.blurV, F.bv, [["T", T.bh]], { px, radius: s.blurRadius });
  exec(
    P.highpass,
    F.hi,
    [
      ["ORIG", T.alb],
      ["BLUR", T.bv],
    ],
    {},
  );

  // Normal → FBO (Curvature reads this)
  exec(
    P.normal,
    F.norm,
    [
      ["DEPTH", T.dep],
      ["DETAIL", T.hi],
    ],
    { px, str: s.normalStr, detailMix: s.detailMix, sigma: s.normalSigma },
  );

  // AO (uses depth)
  exec(P.ao, F.ao, [["T", T.dep]], {
    px,
    radius: s.aoRadius,
    strength: s.aoStr,
  });

  // Roughness (material-aware)
  exec(
    P.rough,
    F.rough,
    [
      ["DEPTH", T.dep],
      ["DETAIL", T.hi],
      ["SEGMASK", T.seg],
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

  // Metallic (material-aware)
  exec(
    P.metal,
    F.metal,
    [
      ["T", T.dep],
      ["SEGMASK", T.seg],
    ],
    { thr: s.metalThr, soft: s.metalSoft },
  );

  // Displacement
  exec(P.displace, F.displace, [["T", T.dep]], {
    mid: s.dispMid,
    con: s.dispCon,
    inv: s.dispInv ? 1 : 0,
  });

  // Snap all to output canvas (sequential: each snap overwrites default FB)
  const snap = (prog, bindings, uniforms) => {
    exec(prog, null, bindings, uniforms);
    return c.toDataURL("image/png");
  };

  return {
    normal: snap(P.pass, [["T", T.norm]], {}),
    displacement: snap(P.displace, [["T", T.dep]], {
      mid: s.dispMid,
      con: s.dispCon,
      inv: s.dispInv ? 1 : 0,
    }),
    roughness: snap(P.pass, [["T", T.rough]], {}),
    metallic: snap(P.pass, [["T", T.metal]], {}),
    ao: snap(P.pass, [["T", T.ao]], {}),
    curvature: snap(P.curv, [["NORMAL", T.norm]], { px, scale: s.curvScale }),
    orm: snap(
      P.orm,
      [
        ["AO", T.ao],
        ["ROUGH", T.rough],
        ["METAL", T.metal],
      ],
      {},
    ),
  };
}

async function generateMaps(
  albedoImg,
  depthCvs,
  segMaskCvs,
  s,
  resolution,
  onProgress,
) {
  const sc = Math.min(
    1,
    resolution / Math.max(albedoImg.naturalWidth, albedoImg.naturalHeight),
  );
  const W = Math.round(albedoImg.naturalWidth * sc);
  const H = Math.round(albedoImg.naturalHeight * sc);

  // Neutral seg mask if not provided
  const neutralSeg =
    segMaskCvs ||
    (() => {
      const c = document.createElement("canvas");
      c.width = c.height = 64;
      const ctx = c.getContext("2d");
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, 64, 64);
      return c;
    })();

  const useTiled = W > TILE_SIZE || H > TILE_SIZE;

  if (!useTiled) {
    // ── Single pass ──────────────────────────────────────────
    const glCtx = buildGLContext(W, H);
    if (!glCtx) return null;
    const albRGBA = toRGBA(albedoImg, W, H);
    const depRGBA = toRGBA(depthCvs, W, H);
    const segRGBA = toRGBA(neutralSeg, W, H);
    onProgress?.(1, 1);
    return runTilePass(glCtx, albRGBA, depRGBA, segRGBA, W, H, s);
  }

  // ── Tiled pass ───────────────────────────────────────────
  const TW = TILE_SIZE + 2 * TILE_PAD;
  const TH = TILE_SIZE + 2 * TILE_PAD;
  const glCtx = buildGLContext(TW, TH);
  if (!glCtx) return null;

  const tilesX = Math.ceil(W / TILE_SIZE);
  const tilesY = Math.ceil(H / TILE_SIZE);
  const totalTiles = tilesX * tilesY;

  // Create full-res scaled source canvases once
  const fullAlb = toRGBA(albedoImg, W, H);
  const fullDep = toRGBA(depthCvs, W, H);
  const fullSeg = toRGBA(neutralSeg, W, H);

  // Output compositing canvases
  const outKeys = [
    "normal",
    "displacement",
    "roughness",
    "metallic",
    "ao",
    "curvature",
    "orm",
  ];
  const outCanvas = {};
  for (const k of outKeys) {
    const c = document.createElement("canvas");
    c.width = W;
    c.height = H;
    outCanvas[k] = c;
  }

  // Tile canvas for source extraction
  const tAlbC = document.createElement("canvas");
  tAlbC.width = tAlbC.height = TW;
  const tDepC = document.createElement("canvas");
  tDepC.width = tDepC.height = TH;
  const tSegC = document.createElement("canvas");
  tSegC.width = tSegC.height = TH;

  let tileNum = 0;
  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const offX = -(tx * TILE_SIZE - TILE_PAD);
      const offY = -(ty * TILE_SIZE - TILE_PAD);

      for (const [tc, src] of [
        [tAlbC, fullAlb],
        [tDepC, fullDep],
        [tSegC, fullSeg],
      ]) {
        const ctx = tc.getContext("2d", { willReadFrequently: true });
        ctx.fillStyle = "#808080";
        ctx.fillRect(0, 0, TW, TH);
        ctx.drawImage(src, offX, offY);
      }

      const result = runTilePass(glCtx, tAlbC, tDepC, tSegC, TW, TH, s);

      // Composite center region (skip padding) to output
      const copyW = Math.min(TILE_SIZE, W - tx * TILE_SIZE);
      const copyH = Math.min(TILE_SIZE, H - ty * TILE_SIZE);
      const dstX = tx * TILE_SIZE;
      const dstY = ty * TILE_SIZE;

      for (const k of outKeys) {
        const tmpImg = new Image();
        tmpImg.src = result[k];
        await new Promise((r) => {
          tmpImg.onload = r;
          tmpImg.onerror = r;
        });
        outCanvas[k]
          .getContext("2d")
          .drawImage(
            tmpImg,
            TILE_PAD,
            TILE_PAD,
            copyW,
            copyH,
            dstX,
            dstY,
            copyW,
            copyH,
          );
      }

      tileNum++;
      onProgress?.(tileNum, totalTiles);
      // Yield to UI thread between tiles
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  const out = {};
  for (const k of outKeys) out[k] = outCanvas[k].toDataURL("image/png");
  return out;
}

/* ═══════════════════════════════════════════════════════════════
   AUTO-ANALYZE  (albedo 통계 → PBR 파라미터 자동 제안)
═══════════════════════════════════════════════════════════════ */
function analyzeAlbedo(imgEl) {
  const S = 128;
  const c = document.createElement("canvas");
  c.width = c.height = S;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(imgEl, 0, 0, S, S);
  const px = ctx.getImageData(0, 0, S, S).data;
  const n = px.length / 4;
  let ss = 0,
    sl = 0,
    lumArr = [];
  for (let i = 0; i < px.length; i += 4) {
    const r = px[i] / 255,
      g = px[i + 1] / 255,
      b = px[i + 2] / 255;
    const mx = Math.max(r, g, b),
      mn = Math.min(r, g, b);
    const l = 0.299 * r + 0.587 * g + 0.114 * b;
    ss += mx === 0 ? 0 : (mx - mn) / mx;
    sl += l;
    lumArr.push(l);
  }
  const avgSat = ss / n,
    avgLum = sl / n;
  const lumVar = lumArr.reduce((a, l) => a + (l - avgLum) ** 2, 0) / n;
  return {
    metalThr: avgSat < 0.12 ? 0.52 : avgSat < 0.25 ? 0.68 : 0.84,
    roughCon: Math.min(5.5, 1.5 + lumVar * 40),
    roughBias: lumVar > 0.04 ? -0.05 : 0.05,
    detailMix: Math.min(0.8, 0.2 + lumVar * 18),
    roughDetailMix: Math.min(0.8, 0.3 + lumVar * 14),
    dispMid: Math.max(0.3, Math.min(0.7, avgLum)),
    matBlend: Math.min(0.5, avgSat < 0.15 ? 0.4 : 0.15),
  };
}

/* ═══════════════════════════════════════════════════════════════
   ZIP EXPORT
═══════════════════════════════════════════════════════════════ */
async function exportZip(maps, resolution) {
  const zip = new JSZip();
  const f = zip.folder(`unarrived_maps_${resolution}px`);
  await Promise.all(
    Object.entries(maps).map(async ([k, u]) => {
      f.file(`unarrived_${k}.png`, await (await fetch(u)).blob());
    }),
  );
  const b = await zip.generateAsync({ type: "blob" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b);
  a.download = `unarrived_maps_${resolution}px.zip`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ═══════════════════════════════════════════════════════════════
   CONSTANTS
═══════════════════════════════════════════════════════════════ */
const DEF = {
  normalStr: 3.0,
  detailMix: 0.4,
  normalSigma: 0.05,
  dispMid: 0.5,
  dispCon: 2.0,
  dispInv: false,
  roughCon: 2.0,
  roughBias: 0.0,
  roughInv: false,
  roughDetailMix: 0.5,
  metalThr: 0.75,
  metalSoft: 0.1,
  blurRadius: 6.0,
  aoRadius: 0.8,
  aoStr: 1.2,
  curvScale: 1.0,
  matBlend: 0.25, // 0=heuristic only  1=class preset only
};

const MAPS_GRID = [
  {
    key: "normal",
    label: "NORMAL",
    color: "#4a90e2",
    hint: "Bilateral depth+highpass",
  },
  {
    key: "displacement",
    label: "DISPLACEMENT",
    color: "#e09a30",
    hint: "Mid-point height field",
  },
  {
    key: "roughness",
    label: "ROUGHNESS",
    color: "#a065d0",
    hint: "Freq-energy + material",
  },
  {
    key: "metallic",
    label: "METALLIC",
    color: "#20b89a",
    hint: "Material-class threshold",
  },
  { key: "ao", label: "AO", color: "#c84b31", hint: "HBAO 32-sample spiral" },
  {
    key: "curvature",
    label: "CURVATURE",
    color: "#f0c040",
    hint: "Multi-scale edge wear",
  },
];
const MAPS_EXTRA = [
  {
    key: "orm",
    label: "ORM PACK",
    color: "#60a0d0",
    hint: "R:AO G:Rough B:Metal (UE/Unity)",
  },
];
const ALL_MAPS = [...MAPS_GRID, ...MAPS_EXTRA];

const AI_STEPS = {
  idle: { label: "IDLE", color: "#1e1e35" },
  loading: { label: "LOADING DEPTH MODEL…", color: "#60a5fa" },
  inferring: { label: "DEPTH INFERENCE…", color: "#22c55e" },
  segloading: { label: "LOADING SEGMENTER…", color: "#a060f0" },
  seginfer: { label: "MATERIAL ANALYSIS…", color: "#d060c0" },
  segmask: { label: "BUILDING MAT MASK…", color: "#d060c0" },
  generating: { label: "BUILDING MAPS…", color: "#e8a020" },
  tiling: { label: "TILING…", color: "#e8a020" },
  analyzing: { label: "ANALYZING IMAGE…", color: "#d06090" },
  ready: { label: "READY", color: "#252540" },
  error: { label: "ERROR", color: "#ef4444" },
};

const SHAPES = [
  { key: "sphere", label: "SPH" },
  { key: "plane", label: "PLN" },
  { key: "torus", label: "TOR" },
];
const HDRI_KEYS = Object.keys(HDRI_PRESETS);
const ALL_RES = [512, 1024, 2048, 4096, 8192];

/* ═══════════════════════════════════════════════════════════════
   APP
═══════════════════════════════════════════════════════════════ */
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
  const [tileProgress, setTileProgress] = useState(null); // {cur,total}
  const [qualityMode, setQualityMode] = useState(false); // false=fast, true=quality
  const [previewShape, setPreviewShape] = useState("sphere");
  const [tileCount, setTileCount] = useState(1);
  const [resolution, setResolution] = useState(1024);
  const [hdriPreset, setHdriPreset] = useState("studio");
  const [gpuInfo, setGpuInfo] = useState({
    maxTex: 4096,
    renderer: "Querying…",
  });
  const [analyzeHint, setAnalyzeHint] = useState(null);

  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const depthRef = useRef(null);
  const segMaskRef = useRef(null);
  const settingRef = useRef(settings);
  const resRef = useRef(resolution);
  const three = useRef({});
  const orbit = useRef({ on: false, x: 0, y: 0 });
  const token = useRef(0);

  useEffect(() => {
    settingRef.current = settings;
  }, [settings]);
  useEffect(() => {
    resRef.current = resolution;
  }, [resolution]);
  useEffect(() => {
    setGpuInfo(queryGPU());
  }, []);

  /* ── Load image & AI inference ─────────────────────────── */
  const loadFile = useCallback(
    (file) => {
      if (!file?.type.startsWith("image/")) return;
      const tid = ++token.current;
      const url = URL.createObjectURL(file);
      const img = new Image();

      img.onload = async () => {
        if (tid !== token.current) return;
        imgRef.current = img;
        setSrcURL(url);
        setImgInfo({ w: img.naturalWidth, h: img.naturalHeight });
        setMaps(null);
        depthRef.current = null;
        segMaskRef.current = null;
        setAnalyzeHint(null);
        setTileProgress(null);
        setProc(true);

        try {
          setAiStep("loading");
          const depEst = await getDepth();
          if (tid !== token.current) return;

          setAiStep("inferring");
          const depResult = await depEst(url);
          if (tid !== token.current) return;
          depthRef.current = depResult.depth.toCanvas();

          // Quality mode: run segmentation
          if (qualityMode) {
            setAiStep("segloading");
            const seg = await getSegmenter();
            if (tid !== token.current) return;

            setAiStep("seginfer");
            const segResult = await seg(url);
            if (tid !== token.current) return;

            setAiStep("segmask");
            segMaskRef.current = await buildMaterialMask(segResult);
          }

          setAiStep("generating");
          await new Promise((r) => requestAnimationFrame(r));
          if (tid !== token.current) return;

          const res = resRef.current;
          const isTiled =
            Math.min(
              1,
              res /
                Math.max(
                  imgRef.current.naturalWidth,
                  imgRef.current.naturalHeight,
                ),
            ) *
              Math.max(
                imgRef.current.naturalWidth,
                imgRef.current.naturalHeight,
              ) >
            TILE_SIZE;
          if (isTiled) setAiStep("tiling");

          const result = await generateMaps(
            imgRef.current,
            depthRef.current,
            segMaskRef.current,
            settingRef.current,
            res,
            (cur, total) => {
              if (tid === token.current) setTileProgress({ cur, total });
            },
          );
          if (tid !== token.current) return;
          setMaps(result);
          setAiStep("ready");
          setTileProgress(null);
        } catch (err) {
          if (tid !== token.current) return;
          console.error("[Mapper]", err);
          setAiStep("error");
        } finally {
          if (tid === token.current) setProc(false);
        }
      };
      img.src = url;
    },
    [qualityMode],
  );

  /* ── Reprocess on settings/resolution change ───────────── */
  useEffect(() => {
    if (!depthRef.current || !imgRef.current) return;
    const id = setTimeout(async () => {
      setProc(true);
      setTileProgress(null);
      const res = resolution;
      const isTiled =
        Math.min(
          1,
          res /
            Math.max(imgRef.current.naturalWidth, imgRef.current.naturalHeight),
        ) *
          Math.max(imgRef.current.naturalWidth, imgRef.current.naturalHeight) >
        TILE_SIZE;
      if (isTiled) setAiStep("tiling");
      else setAiStep("generating");
      const result = await generateMaps(
        imgRef.current,
        depthRef.current,
        segMaskRef.current,
        settings,
        res,
        (c, t) => setTileProgress({ cur: c, total: t }),
      );
      setMaps(result);
      setAiStep("ready");
      setTileProgress(null);
      setProc(false);
    }, 250);
    return () => clearTimeout(id);
  }, [settings, resolution]);

  /* ── Three.js init ─────────────────────────────────────── */
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const renderer = new THREE.WebGLRenderer({ canvas: el, antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    renderer.outputEncoding = THREE.sRGBEncoding;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 50);
    camera.position.z = 3.2;
    const env = buildEnvMap(renderer, "studio");
    scene.environment = env;
    scene.background = env;
    const key = new THREE.DirectionalLight(0xfff4d6, 0.6);
    key.position.set(4, 5, 3);
    scene.add(key);
    const geos = {
      sphere: new THREE.SphereGeometry(1, 128, 64),
      plane: new THREE.PlaneGeometry(2, 2, 64, 64),
      torus: new THREE.TorusGeometry(0.72, 0.34, 64, 128),
    };
    Object.values(geos).forEach((g) => g.setAttribute("uv2", g.attributes.uv));
    const mat = new THREE.MeshStandardMaterial({
      roughness: 0.5,
      metalness: 0.0,
      color: 0xcccccc,
      envMapIntensity: 1,
    });
    const mesh = new THREE.Mesh(geos.sphere, mat);
    scene.add(mesh);
    const gMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(20, 20),
      new THREE.MeshStandardMaterial({
        color: 0x0a0a20,
        roughness: 1,
        metalness: 0,
      }),
    );
    gMesh.rotation.x = -Math.PI / 2;
    gMesh.position.y = -1.4;
    scene.add(gMesh);
    three.current = { renderer, scene, camera, mat, mesh, geos };
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
    const { renderer, scene } = three.current;
    if (!renderer || !scene) return;
    const e = buildEnvMap(renderer, hdriPreset);
    scene.environment = e;
    scene.background = e;
  }, [hdriPreset]);
  useEffect(() => {
    const { mesh, geos } = three.current;
    if (!mesh) return;
    mesh.geometry = geos[previewShape] ?? geos.sphere;
  }, [previewShape]);
  useEffect(() => {
    const { mat } = three.current;
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
      t.repeat.set(tileCount, tileCount);
      t.needsUpdate = true;
    });
  }, [tileCount]);
  useEffect(() => {
    const { mat } = three.current;
    if (!mat || !srcURL) return;
    new THREE.TextureLoader().load(srcURL, (t) => {
      t.colorSpace = THREE.SRGBColorSpace;
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.repeat.set(tileCount, tileCount);
      mat.map?.dispose();
      mat.map = t;
      mat.needsUpdate = true;
    });
  }, [srcURL]);
  useEffect(() => {
    const { mat } = three.current;
    if (!mat || !maps) return;
    const L = new THREE.TextureLoader();
    const ld = (u, cb) =>
      L.load(u, (t) => {
        t.wrapS = t.wrapT = THREE.RepeatWrapping;
        t.repeat.set(tileCount, tileCount);
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

  /* ── Helpers ────────────────────────────────────────────── */
  const setS = (k) => (e) =>
    setSettings((s) => ({ ...s, [k]: parseFloat(e.target.value) }));
  const togS = (k) => () => setSettings((s) => ({ ...s, [k]: !s[k] }));
  const dl = (key) => {
    if (!maps?.[key]) return;
    const a = document.createElement("a");
    a.href = maps[key];
    a.download = `unarrived_${key}.png`;
    a.click();
  };
  const dlAll = async () => {
    if (!maps || exporting) return;
    setExporting(true);
    try {
      await exportZip(maps, resolution);
    } finally {
      setExporting(false);
    }
  };
  const runAnalyze = useCallback(() => {
    if (!imgRef.current || proc) return;
    setAiStep("analyzing");
    setTimeout(() => {
      const h = analyzeAlbedo(imgRef.current);
      setAnalyzeHint(h);
      setSettings((s) => ({ ...s, ...h }));
      setAiStep("ready");
    }, 80);
  }, [proc]);

  const step = AI_STEPS[aiStep];
  const availRes = ALL_RES.filter((r) => r <= gpuInfo.maxTex);
  const tileTotal = tileProgress?.total ?? 0;
  const tileCur = tileProgress?.cur ?? 0;

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Oxanium:wght@300;400;600;700&family=JetBrains+Mono:wght@300;400&display=swap');
        *,*::before,*::after{box-sizing:border-box;margin:0;padding:0} body{overflow:hidden}
        ::-webkit-scrollbar{width:3px} ::-webkit-scrollbar-track{background:#06060f} ::-webkit-scrollbar-thumb{background:#1a1a2e;border-radius:2px}
        input[type=range]{-webkit-appearance:none;width:100%;height:2px;background:#181828;outline:none;cursor:pointer;border-radius:1px}
        input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:11px;height:11px;border-radius:50%;background:#e8a020;cursor:pointer;box-shadow:0 0 6px #e8a02055}
        @keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
        @keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
        @keyframes pulse{0%,100%{opacity:.6}50%{opacity:1}}
        @keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
      `}</style>

      <div
        style={{
          display: "flex",
          height: "100vh",
          background: "#060610",
          color: "#9090b8",
          fontFamily: "'Oxanium',sans-serif",
          overflow: "hidden",
          userSelect: "none",
        }}
      >
        {/* ═══ LEFT SIDEBAR ═══ */}
        <div
          style={{
            width: 252,
            flexShrink: 0,
            borderRight: "1px solid #111124",
            display: "flex",
            flexDirection: "column",
            overflowY: "auto",
          }}
        >
          {/* Logo */}
          <div
            style={{
              padding: "13px 18px 9px",
              borderBottom: "1px solid #111124",
            }}
          >
            <div
              style={{
                fontSize: 6,
                letterSpacing: 6,
                color: "#20203a",
                marginBottom: 3,
              }}
            >
              UNARRIVED
            </div>
            <div
              style={{
                fontSize: 19,
                fontWeight: 700,
                color: "#e8a020",
                letterSpacing: 2,
                lineHeight: 1,
              }}
            >
              MAPPER
            </div>
            <div
              style={{
                fontSize: 6,
                color: "#22c55e",
                letterSpacing: 3,
                marginTop: 4,
                fontFamily: "'JetBrains Mono',monospace",
              }}
            >
              SMART PBR · MATERIAL AI · v6
            </div>
          </div>

          {/* Drop zone */}
          <div style={{ padding: "9px 9px 5px" }}>
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
                border: `1px dashed ${dragOver ? "#e8a020" : "#1c1c30"}`,
                borderRadius: 5,
                cursor: "pointer",
                background: dragOver ? "#120e01" : "#09091a",
                transition: "all .2s",
                overflow: "hidden",
              }}
            >
              {srcURL ? (
                <img
                  src={srcURL}
                  style={{
                    width: "100%",
                    height: 68,
                    objectFit: "cover",
                    display: "block",
                  }}
                />
              ) : (
                <div style={{ padding: "16px 12px", textAlign: "center" }}>
                  <div
                    style={{ fontSize: 18, color: "#18182e", marginBottom: 4 }}
                  >
                    ⬡
                  </div>
                  <div
                    style={{ fontSize: 7, letterSpacing: 3, color: "#30304a" }}
                  >
                    DROP IMAGE HERE
                  </div>
                </div>
              )}
              <div
                style={{
                  padding: "3px 9px",
                  fontSize: 7,
                  letterSpacing: 3,
                  color: srcURL ? "#303050" : "#202035",
                  background: "#07070f",
                }}
              >
                {srcURL ? "ALBEDO · REPLACE" : "OR BROWSE"}
              </div>
            </div>
          </div>

          {/* Mode + Status */}
          <div
            style={{
              padding: "5px 10px 4px",
              borderBottom: "1px solid #0d0d1e",
            }}
          >
            {/* Quality mode toggle */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: 5,
              }}
            >
              <div>
                <div
                  style={{
                    fontSize: 6,
                    letterSpacing: 3,
                    color: qualityMode ? "#d060c0" : "#282840",
                    fontFamily: "'JetBrains Mono',monospace",
                  }}
                >
                  {qualityMode ? "QUALITY MODE" : "FAST MODE"}
                </div>
                <div
                  style={{
                    fontSize: 5,
                    letterSpacing: 2,
                    color: "#141428",
                    fontFamily: "'JetBrains Mono',monospace",
                    marginTop: 1,
                  }}
                >
                  {qualityMode
                    ? "+ Material Segmentation"
                    : "Depth-Anything V2 only"}
                </div>
              </div>
              <div
                onClick={() => setQualityMode((v) => !v)}
                style={{
                  width: 30,
                  height: 14,
                  borderRadius: 7,
                  cursor: "pointer",
                  position: "relative",
                  background: qualityMode ? "#8020a0" : "#141428",
                  transition: "background .2s",
                  border: "1px solid #2a1a3a",
                  flexShrink: 0,
                }}
              >
                <div
                  style={{
                    position: "absolute",
                    top: 2,
                    width: 9,
                    height: 9,
                    borderRadius: 5,
                    background: qualityMode ? "#f090ff" : "#383858",
                    left: qualityMode ? 19 : 2,
                    transition: "left .2s",
                  }}
                />
              </div>
            </div>
            {/* Status row */}
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span
                style={{
                  fontSize: 8,
                  color: step.color,
                  animation: proc ? "blink 1s infinite" : "none",
                }}
              >
                ◈
              </span>
              <span
                style={{
                  fontSize: 7,
                  letterSpacing: 2,
                  color: step.color,
                  fontFamily: "'JetBrains Mono',monospace",
                  flex: 1,
                  animation: proc ? "pulse 1.4s infinite" : "none",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {step.label}
              </span>
              <button
                onClick={runAnalyze}
                disabled={!imgRef.current || proc}
                style={{
                  padding: "2px 7px",
                  borderRadius: 2,
                  fontSize: 6,
                  letterSpacing: 1,
                  cursor: imgRef.current && !proc ? "pointer" : "not-allowed",
                  fontFamily: "'JetBrains Mono',monospace",
                  background: imgRef.current && !proc ? "#180a14" : "#08080f",
                  border: `1px solid ${imgRef.current && !proc ? "#501840" : "#0d0d1e"}`,
                  color: imgRef.current && !proc ? "#c050a0" : "#1e1e30",
                  whiteSpace: "nowrap",
                  flexShrink: 0,
                }}
              >
                AI ANA
              </button>
            </div>
            {/* Tile progress bar */}
            {tileProgress && (
              <div style={{ marginTop: 5 }}>
                <div
                  style={{
                    fontSize: 6,
                    letterSpacing: 2,
                    color: "#e8a02066",
                    fontFamily: "'JetBrains Mono',monospace",
                    marginBottom: 3,
                  }}
                >
                  TILE {tileCur}/{tileTotal}
                </div>
                <div
                  style={{ height: 2, background: "#181828", borderRadius: 1 }}
                >
                  <div
                    style={{
                      height: "100%",
                      background: "#e8a020",
                      borderRadius: 1,
                      width: `${((tileCur / tileTotal) * 100).toFixed(0)}%`,
                      transition: "width .2s",
                    }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Auto-analyze result */}
          {analyzeHint && (
            <div
              style={{
                padding: "5px 11px",
                background: "#0d0820",
                borderBottom: "1px solid #1a0830",
              }}
            >
              <div
                style={{
                  fontSize: 6,
                  letterSpacing: 2,
                  color: "#a040a0",
                  fontFamily: "'JetBrains Mono',monospace",
                  marginBottom: 3,
                }}
              >
                ◈ ANALYZED
              </div>
              <div
                style={{
                  fontSize: 5,
                  color: "#5a1a5a",
                  fontFamily: "'JetBrains Mono',monospace",
                  lineHeight: 1.6,
                }}
              >
                {Object.entries(analyzeHint).map(([k, v]) => (
                  <span key={k} style={{ marginRight: 8 }}>
                    {k}={typeof v === "number" ? v.toFixed(2) : String(v)}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Controls */}
          <div style={{ padding: "0 12px 10px", flex: 1 }}>
            <Sec label="NORMAL MAP">
              <Sld
                label="DEPTH STR"
                v={settings.normalStr}
                min={0.5}
                max={14}
                step={0.1}
                onChange={setS("normalStr")}
              />
              <Sld
                label="DETAIL MIX"
                v={settings.detailMix}
                min={0}
                max={1}
                step={0.01}
                onChange={setS("detailMix")}
                hint="Highpass → normal blend"
              />
              <Sld
                label="BILATERAL σ"
                v={settings.normalSigma}
                min={0.01}
                max={0.3}
                step={0.01}
                onChange={setS("normalSigma")}
                hint="Depth edge preservation"
              />
            </Sec>

            <Sec label="DISPLACEMENT">
              <Sld
                label="MIDPOINT"
                v={settings.dispMid}
                min={0}
                max={1}
                step={0.01}
                onChange={setS("dispMid")}
                hint="Neutral depth level"
              />
              <Sld
                label="CONTRAST"
                v={settings.dispCon}
                min={0.5}
                max={6}
                step={0.1}
                onChange={setS("dispCon")}
              />
              <Tog
                label="INVERT"
                v={settings.dispInv}
                onChange={togS("dispInv")}
              />
            </Sec>

            <Sec label="ROUGHNESS">
              <Sld
                label="CONTRAST"
                v={settings.roughCon}
                min={0.5}
                max={6}
                step={0.1}
                onChange={setS("roughCon")}
              />
              <Sld
                label="BIAS"
                v={settings.roughBias}
                min={-0.5}
                max={0.5}
                step={0.01}
                onChange={setS("roughBias")}
              />
              <Sld
                label="FREQ MIX"
                v={settings.roughDetailMix}
                min={0}
                max={1}
                step={0.01}
                onChange={setS("roughDetailMix")}
                hint="0=depth · 1=freq-energy"
              />
              <Sld
                label="MAT BLEND"
                v={settings.matBlend}
                min={0}
                max={1}
                step={0.01}
                onChange={setS("matBlend")}
                hint="0=heuristic · 1=class preset"
              />
              <Tog
                label="INVERT"
                v={settings.roughInv}
                onChange={togS("roughInv")}
              />
            </Sec>

            <Sec label="METALLIC">
              <Sld
                label="THRESHOLD"
                v={settings.metalThr}
                min={0}
                max={1}
                step={0.01}
                onChange={setS("metalThr")}
              />
              <Sld
                label="SOFTNESS"
                v={settings.metalSoft}
                min={0.01}
                max={0.3}
                step={0.01}
                onChange={setS("metalSoft")}
              />
            </Sec>

            <Sec label="HBAO">
              <Sld
                label="RADIUS"
                v={settings.aoRadius}
                min={0.1}
                max={3}
                step={0.05}
                onChange={setS("aoRadius")}
              />
              <Sld
                label="STRENGTH"
                v={settings.aoStr}
                min={0}
                max={3}
                step={0.05}
                onChange={setS("aoStr")}
              />
            </Sec>

            <Sec label="CURVATURE">
              <Sld
                label="SCALE"
                v={settings.curvScale}
                min={0.1}
                max={4}
                step={0.05}
                onChange={setS("curvScale")}
                hint="Multi-scale blend: 1px / 4px / 12px"
              />
            </Sec>

            <Sec label="FREQ SEPARATION">
              <Sld
                label="BLUR RADIUS"
                v={settings.blurRadius}
                min={1}
                max={24}
                step={0.5}
                onChange={setS("blurRadius")}
              />
            </Sec>
          </div>

          {/* GPU + Resolution */}
          <div style={{ padding: "6px 10px", borderTop: "1px solid #111124" }}>
            <div
              style={{
                fontSize: 5,
                letterSpacing: 2,
                color: "#0c0c1c",
                fontFamily: "'JetBrains Mono',monospace",
                marginBottom: 4,
                lineHeight: 1.7,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {gpuInfo.renderer} · MAX {gpuInfo.maxTex}px
            </div>
            <div
              style={{
                fontSize: 6,
                letterSpacing: 4,
                color: "#1a1a30",
                marginBottom: 5,
              }}
            >
              OUTPUT RES
            </div>
            <div style={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
              {ALL_RES.map((r) => {
                const avail = r <= gpuInfo.maxTex;
                const active = resolution === r;
                const warn = r >= 4096 && avail;
                return (
                  <button
                    key={r}
                    onClick={() => avail && setResolution(r)}
                    disabled={!avail}
                    style={{
                      flex: "1 0 38px",
                      padding: "4px 2px",
                      borderRadius: 2,
                      fontSize: 7,
                      cursor: avail ? "pointer" : "not-allowed",
                      fontFamily: "'JetBrains Mono',monospace",
                      transition: "all .15s",
                      background: active
                        ? "#1a1a2e"
                        : avail
                          ? "#08081a"
                          : "#050508",
                      border: `1px solid ${active ? "#4a4a70" : warn ? "#403010" : avail ? "#111128" : "#090912"}`,
                      color: active
                        ? "#c8c8e8"
                        : warn
                          ? "#806020"
                          : avail
                            ? "#282840"
                            : "#141418",
                      opacity: avail ? 1 : 0.4,
                    }}
                  >
                    {r >= 1000 ? `${r / 1000}K` : r}
                  </button>
                );
              })}
            </div>
            {resolution >= 4096 && (
              <div
                style={{
                  marginTop: 4,
                  fontSize: 5,
                  color: "#504010",
                  fontFamily: "'JetBrains Mono',monospace",
                  lineHeight: 1.6,
                }}
              >
                ⚠ Tiled · ~{vramMB(TILE_SIZE)}MB VRAM/tile
              </div>
            )}
          </div>

          {/* Export */}
          <div
            style={{ padding: "7px 9px 11px", borderTop: "1px solid #111124" }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 5,
              }}
            >
              <div style={{ fontSize: 6, letterSpacing: 4, color: "#1e1e35" }}>
                EXPORT
              </div>
              <button
                onClick={dlAll}
                disabled={!maps || exporting}
                style={{
                  padding: "2px 8px",
                  borderRadius: 2,
                  fontSize: 7,
                  letterSpacing: 2,
                  cursor: maps && !exporting ? "pointer" : "not-allowed",
                  fontFamily: "'JetBrains Mono',monospace",
                  background: maps && !exporting ? "#1a2030" : "#080810",
                  border: `1px solid ${maps && !exporting ? "#2a4060" : "#0d0d1e"}`,
                  color: maps && !exporting ? "#60a0e0" : "#1e1e30",
                  display: "flex",
                  alignItems: "center",
                  gap: 4,
                }}
              >
                {exporting ? (
                  <span
                    style={{
                      animation: "spin 1s linear infinite",
                      display: "inline-block",
                    }}
                  >
                    ⟳
                  </span>
                ) : (
                  <span>⬡</span>
                )}
                <span>{exporting ? "…" : "ZIP ALL"}</span>
              </button>
            </div>
            {ALL_MAPS.map((m) => (
              <button
                key={m.key}
                onClick={() => dl(m.key)}
                disabled={!maps?.[m.key]}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 5,
                  width: "100%",
                  padding: "4px 7px",
                  marginBottom: 2,
                  borderRadius: 3,
                  background: "#0a0a1a",
                  border: `1px solid ${maps?.[m.key] ? "#1c1c32" : "#0d0d1e"}`,
                  color: maps?.[m.key] ? m.color : "#1e1e30",
                  cursor: maps?.[m.key] ? "pointer" : "not-allowed",
                  fontFamily: "'JetBrains Mono',monospace",
                  fontSize: 7,
                  letterSpacing: 2,
                  textAlign: "left",
                  transition: "all .15s",
                }}
              >
                <span
                  style={{ opacity: maps?.[m.key] ? 0.7 : 0.3, fontSize: 8 }}
                >
                  ↓
                </span>
                <span>{m.label}</span>
                {m.key === "orm" && (
                  <span
                    style={{
                      fontSize: 5,
                      color: maps?.[m.key] ? "#304050" : "#141420",
                      marginLeft: "auto",
                    }}
                  >
                    UE/UNITY
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* ═══ CENTER: 3×2 MAP GRID ═══ */}
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            minWidth: 0,
          }}
        >
          <div
            style={{
              padding: "5px 13px",
              borderBottom: "1px solid #111124",
              display: "flex",
              gap: 12,
              alignItems: "center",
              fontFamily: "'JetBrains Mono',monospace",
              fontSize: 7,
              letterSpacing: 2,
              color: "#1e1e35",
              flexShrink: 0,
            }}
          >
            <span style={{ color: proc ? step.color + "88" : "#1e1e35" }}>
              {tileProgress ? `TILE ${tileCur}/${tileTotal}` : step.label}
            </span>
            {imgInfo && (
              <span>
                SRC {imgInfo.w}×{imgInfo.h}
              </span>
            )}
            <span>OUT {resolution}px</span>
            <span style={{ marginLeft: "auto", color: "#0e0e1e" }}>
              {qualityMode ? "MAT-SEG" : "DEPTH-V2"} · BILATERAL · HBAO · ORM
            </span>
          </div>
          <div
            style={{
              flex: 1,
              display: "grid",
              gridTemplateColumns: "1fr 1fr 1fr",
              gridTemplateRows: "1fr 1fr",
              gap: 1,
              background: "#020208",
              overflow: "hidden",
            }}
          >
            {MAPS_GRID.map((m) => (
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
        </div>

        {/* ═══ RIGHT: 3D PREVIEW ═══ */}
        <div
          style={{
            width: 258,
            flexShrink: 0,
            borderLeft: "1px solid #111124",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div
            style={{
              padding: "5px 12px",
              borderBottom: "1px solid #111124",
              fontSize: 6,
              letterSpacing: 5,
              color: "#1a1a30",
              fontFamily: "'JetBrains Mono',monospace",
              flexShrink: 0,
            }}
          >
            PBR PREVIEW
          </div>

          <div style={{ flex: 1, position: "relative" }}>
            <canvas
              ref={canvasRef}
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
                <div
                  style={{
                    textAlign: "center",
                    fontSize: 7,
                    letterSpacing: 4,
                    color: "#141428",
                    fontFamily: "'JetBrains Mono',monospace",
                    lineHeight: 2,
                  }}
                >
                  LOAD IMAGE
                  <br />
                  TO PREVIEW
                </div>
              </div>
            )}
          </div>

          {/* Shape */}
          <div
            style={{
              padding: "5px 9px",
              borderTop: "1px solid #111124",
              display: "flex",
              gap: 3,
              flexShrink: 0,
            }}
          >
            {SHAPES.map((sh) => (
              <button
                key={sh.key}
                onClick={() => setPreviewShape(sh.key)}
                style={{
                  flex: 1,
                  padding: "3px 0",
                  borderRadius: 2,
                  fontSize: 7,
                  letterSpacing: 2,
                  cursor: "pointer",
                  fontFamily: "'JetBrains Mono',monospace",
                  transition: "all .15s",
                  background: previewShape === sh.key ? "#1a1a2e" : "#08081a",
                  border: `1px solid ${previewShape === sh.key ? "#3a3a60" : "#111128"}`,
                  color: previewShape === sh.key ? "#9090c8" : "#282840",
                }}
              >
                {sh.label}
              </button>
            ))}
          </div>

          {/* Tile */}
          <div
            style={{
              padding: "4px 9px",
              borderTop: "1px solid #111124",
              display: "flex",
              gap: 3,
              alignItems: "center",
              flexShrink: 0,
            }}
          >
            <span
              style={{
                fontSize: 6,
                letterSpacing: 3,
                color: "#1a1a30",
                fontFamily: "'JetBrains Mono',monospace",
                flexShrink: 0,
              }}
            >
              TILE
            </span>
            {[1, 2, 4].map((n) => (
              <button
                key={n}
                onClick={() => setTileCount(n)}
                style={{
                  flex: 1,
                  padding: "2px 0",
                  borderRadius: 2,
                  fontSize: 8,
                  cursor: "pointer",
                  fontFamily: "'JetBrains Mono',monospace",
                  background: tileCount === n ? "#1a1a2e" : "#08081a",
                  border: `1px solid ${tileCount === n ? "#3a3a60" : "#111128"}`,
                  color: tileCount === n ? "#9090c8" : "#282840",
                }}
              >
                {n}×
              </button>
            ))}
          </div>

          {/* HDRI */}
          <div
            style={{
              padding: "5px 9px",
              borderTop: "1px solid #111124",
              flexShrink: 0,
            }}
          >
            <div
              style={{
                fontSize: 6,
                letterSpacing: 4,
                color: "#141428",
                fontFamily: "'JetBrains Mono',monospace",
                marginBottom: 4,
              }}
            >
              ENV
            </div>
            <div style={{ display: "flex", gap: 3 }}>
              {HDRI_KEYS.map((k) => (
                <button
                  key={k}
                  onClick={() => setHdriPreset(k)}
                  style={{
                    flex: 1,
                    padding: "3px 0",
                    borderRadius: 2,
                    fontSize: 6,
                    letterSpacing: 1,
                    cursor: "pointer",
                    fontFamily: "'JetBrains Mono',monospace",
                    background: hdriPreset === k ? "#1a1a2e" : "#08081a",
                    border: `1px solid ${hdriPreset === k ? "#3a3a60" : "#111128"}`,
                    color: hdriPreset === k ? "#9090c8" : "#202038",
                  }}
                >
                  {HDRI_PRESETS[k].label}
                </button>
              ))}
            </div>
          </div>

          <div
            style={{
              padding: "4px 9px",
              borderTop: "1px solid #111124",
              fontSize: 5,
              color: "#0e0e1e",
              letterSpacing: 3,
              fontFamily: "'JetBrains Mono',monospace",
              flexShrink: 0,
            }}
          >
            DRAG ORBIT · AUTO-ROTATE
          </div>

          {/* Map legend */}
          <div
            style={{
              padding: "5px 9px 7px",
              borderTop: "1px solid #111124",
              flexShrink: 0,
            }}
          >
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "2px 6px",
              }}
            >
              {ALL_MAPS.map((m) => (
                <div
                  key={m.key}
                  style={{ display: "flex", alignItems: "center", gap: 4 }}
                >
                  <div
                    style={{
                      width: 4,
                      height: 4,
                      borderRadius: 1,
                      background: m.color,
                      opacity: 0.7,
                      flexShrink: 0,
                    }}
                  />
                  <span
                    style={{
                      fontSize: 5,
                      letterSpacing: 2,
                      color: "#141428",
                      fontFamily: "'JetBrains Mono',monospace",
                    }}
                  >
                    {m.label}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Pipeline info */}
          <div
            style={{
              padding: "5px 9px 8px",
              borderTop: "1px solid #111124",
              flexShrink: 0,
            }}
          >
            <div
              style={{
                fontSize: 5,
                letterSpacing: 2,
                color: "#0c0c1c",
                fontFamily: "'JetBrains Mono',monospace",
                lineHeight: 1.8,
              }}
            >
              DEPTH · depth-anything-v2
              <br />
              SEG · segformer-b0-ade (quality)
              <br />
              NORM · bilateral sobel
              <br />
              AO · HBAO 32-sample spiral
              <br />
              CURV · multi-scale 1/4/12px
              <br />
              8K · tiled 2048×N + PAD 32
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

/* ═══════════════════════════════════════════════════════════════
   SUB-COMPONENTS
═══════════════════════════════════════════════════════════════ */
function Sec({ label, children }) {
  return (
    <div
      style={{
        marginBottom: 11,
        paddingTop: 10,
        borderTop: "1px solid #0e0e1e",
      }}
    >
      <div
        style={{
          fontSize: 6,
          letterSpacing: 4,
          color: "#242440",
          marginBottom: 8,
        }}
      >
        {label}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
        {children}
      </div>
    </div>
  );
}
function Sld({ label, v, min, max, step, onChange, hint }) {
  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 4,
        }}
      >
        <span style={{ fontSize: 7, letterSpacing: 3, color: "#383858" }}>
          {label}
        </span>
        <span
          style={{
            fontSize: 9,
            fontFamily: "'JetBrains Mono',monospace",
            color: "#e8a020",
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
        <div
          style={{
            fontSize: 5,
            letterSpacing: 1,
            color: "#141428",
            fontFamily: "'JetBrains Mono',monospace",
            marginTop: 2,
          }}
        >
          {hint}
        </div>
      )}
    </div>
  );
}
function Tog({ label, v, onChange }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}
    >
      <span style={{ fontSize: 7, letterSpacing: 3, color: "#383858" }}>
        {label}
      </span>
      <div
        onClick={onChange}
        style={{
          width: 28,
          height: 13,
          borderRadius: 7,
          cursor: "pointer",
          position: "relative",
          background: v ? "#e8a020" : "#141428",
          transition: "background .2s",
          border: "1px solid #1e1e38",
        }}
      >
        <div
          style={{
            position: "absolute",
            top: 1.5,
            width: 9,
            height: 9,
            borderRadius: 5,
            background: v ? "#070710" : "#383858",
            left: v ? 16 : 2,
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
        background: "#07070f",
        position: "relative",
        overflow: "hidden",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
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
            animation: "fadeIn .3s ease",
          }}
        />
      ) : (
        <div style={{ textAlign: "center", pointerEvents: "none" }}>
          <div
            style={{
              fontSize: 7,
              letterSpacing: 4,
              color: "#111125",
              fontFamily: "'JetBrains Mono',monospace",
              lineHeight: 2,
            }}
          >
            {meta.label}
            <br />
            <span style={{ color: "#0c0c1e" }}>AWAITING AI</span>
          </div>
        </div>
      )}
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          padding: "4px 7px",
          background: "linear-gradient(to bottom,rgba(6,6,16,.95),transparent)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span
          style={{
            fontSize: 7,
            letterSpacing: 3,
            color: meta.color,
            opacity: 0.9,
            fontFamily: "'JetBrains Mono',monospace",
          }}
        >
          {meta.label}
        </span>
        {url && hovered && (
          <button
            onClick={onDl}
            style={{
              background: "none",
              border: `1px solid ${meta.color}55`,
              color: meta.color,
              fontSize: 7,
              letterSpacing: 2,
              padding: "2px 6px",
              cursor: "pointer",
              borderRadius: 2,
              fontFamily: "'JetBrains Mono',monospace",
              opacity: 0.85,
            }}
          >
            ↓
          </button>
        )}
      </div>
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          padding: "3px 7px",
          background: "linear-gradient(to top,rgba(6,6,16,.9),transparent)",
          fontSize: 5,
          color: "#181828",
          letterSpacing: 2,
          fontFamily: "'JetBrains Mono',monospace",
        }}
      >
        {meta.hint}
      </div>
    </div>
  );
}
