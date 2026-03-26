import { useState, useRef, useEffect, useCallback } from "react";
import * as THREE from "three";
import { GLTFExporter } from "three/examples/jsm/exporters/GLTFExporter.js";
import { pipeline } from "@huggingface/transformers";
import JSZip from "jszip";

/* ═══════════════════════════════════════════════════════════════
   1. AI SINGLETONS & DEVICE FALLBACK
═══════════════════════════════════════════════════════════════ */
const DEPTH_MODEL = "onnx-community/depth-anything-v2-small";
const SEG_MODEL_FAST = "Xenova/segformer-b0-finetuned-ade-512-512";
const SEG_MODEL_HQ = "Xenova/segformer-b2-finetuned-ade-512-512";

let _depthP = null,
  _segP = null,
  _segHQP = null;
const inferDevice = () =>
  typeof navigator !== "undefined" && navigator.gpu ? "webgpu" : "wasm";

const makePipe = (task, model) =>
  pipeline(task, model, { device: inferDevice() }).catch(async (err) => {
    if (inferDevice() === "webgpu") {
      console.warn(`[AI] WebGPU failed for ${model}, falling back to WASM.`);
      try {
        return await pipeline(task, model, { device: "wasm" });
      } catch (_) {}
    }
    throw err;
  });

const getDepth = () => {
  if (!_depthP)
    _depthP = makePipe("depth-estimation", DEPTH_MODEL).catch((e) => {
      _depthP = null;
      throw e;
    });
  return _depthP;
};
const getSeg = (hq = false) => {
  if (hq) {
    if (!_segHQP)
      _segHQP = makePipe("image-segmentation", SEG_MODEL_HQ).catch((e) => {
        _segHQP = null;
        throw e;
      });
    return _segHQP;
  }
  if (!_segP)
    _segP = makePipe("image-segmentation", SEG_MODEL_FAST).catch((e) => {
      _segP = null;
      throw e;
    });
  return _segP;
};

function getSafeImage(img, maxDim = 1024) {
  const w = img.naturalWidth,
    h = img.naturalHeight;
  if (w <= maxDim && h <= maxDim) return img.src;
  const sc = maxDim / Math.max(w, h);
  const c = document.createElement("canvas");
  c.width = Math.round(w * sc);
  c.height = Math.round(h * sc);
  c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.8);
}

function queryGPU() {
  try {
    const c = document.createElement("canvas"),
      gl = c.getContext("webgl2");
    if (!gl) return { maxTex: 2048, renderer: "WebGL2 N/A" };
    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE),
      ext = gl.getExtension("WEBGL_debug_renderer_info");
    const renderer = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : "GPU";
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return { maxTex, renderer };
  } catch {
    return { maxTex: 2048, renderer: "N/A" };
  }
}

/* ═══════════════════════════════════════════════════════════════
   2. PROCEDURAL HDRI & DECOUPLED MATERIAL PROFILES
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
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.flipY = false;
  const pm = new THREE.PMREMGenerator(renderer);
  pm.compileEquirectangularShader();
  const env = pm.fromEquirectangular(tex).texture;
  pm.dispose();
  tex.dispose();
  return env;
}

const MAT_PROFILES = {
  rough: [0.75, 0.38, 0.86, 0.92, 0.22],
  cc: [0.0, 1.0, 0.0, 0.0, 0.5],
  sheen: [0.0, 0.0, 0.0, 1.0, 0.0],
  trans: [0.0, 0.0, 0.0, 0.0, 0.9],
};

const LABEL_MAP = {
  car: 1,
  bus: 1,
  truck: 1,
  bicycle: 1,
  motorcycle: 1,
  steel: 1,
  iron: 1,
  metal: 1,
  aluminum: 1,
  chrome: 1,
  wall: 2,
  floor: 2,
  road: 2,
  pavement: 2,
  rock: 2,
  stone: 2,
  concrete: 2,
  building: 2,
  brick: 2,
  tile: 2,
  ceramic: 2,
  wood: 2,
  curtain: 3,
  sofa: 3,
  bed: 3,
  fabric: 3,
  cloth: 3,
  rug: 3,
  carpet: 3,
  cushion: 3,
  pillow: 3,
  paper: 3,
  cardboard: 3,
  grass: 3,
  leaf: 3,
  tree: 3,
  plant: 3,
  bottle: 4,
  glass: 4,
  window: 4,
  plastic: 4,
  jar: 4,
  water: 4,
  vase: 4,
  cup: 4,
};
const labelCls = (l) => {
  const low = String(l || "").toLowerCase();
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
    const enc = Math.round((labelCls(seg.label || "") / 4) * 255);
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
   3. GLSL SHADERS
═══════════════════════════════════════════════════════════════ */
const GLSL_COMMON = `
  float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233))) * 43758.5453); }
  vec2 hash2(vec2 p){ p=vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))); return fract(sin(p)*43758.5453); }
  float noise(vec2 p){ vec2 i=floor(p); vec2 f=fract(p); vec2 u=f*f*(3.0-2.0*f); return mix(mix(hash(i),hash(i+vec2(1,0)),u.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),u.x),u.y); } 
  float fbm(vec2 p){ float v=0.,a=0.5; vec2 shift=vec2(100.); mat2 rot=mat2(cos(0.5),sin(0.5),-sin(0.5),cos(0.5)); for(int i=0;i<5;++i){ v+=a*noise(p); p=rot*p*2.+shift; a*=0.5; } return v; }
  float voronoi(vec2 x) { vec2 n=floor(x),f=fract(x); float m=8.0; for(int j=-1;j<=1;j++)for(int i=-1;i<=1;i++){ vec2 g=vec2(float(i),float(j)),o=hash2(n+g),r=g-f+o; m=min(m,dot(r,r)); } return sqrt(m); }
  float scratch(vec2 uv, float angle, float density) { float s=sin(angle),c=cos(angle); vec2 r=vec2(uv.x*c-uv.y*s,uv.x*s+uv.y*c); r.x*=density; r.y*=2.0; return smoothstep(0.85,0.95,noise(r*20.0)); }
`;

const VS = `#version 300 es\nin vec2 a;out vec2 v;\nvoid main(){v=a*.5+.5;gl_Position=vec4(a,0.,1.);}`;
const FS_PASS = `#version 300 es\nprecision highp float;\nuniform sampler2D T;in vec2 v;out vec4 o;\nvoid main(){o=texture(T,v);}`;
const FS_CHANNEL = `#version 300 es\nprecision highp float; uniform sampler2D T; uniform float ch; in vec2 v; out vec4 o; void main(){ float c=0.0; if(ch<0.5) c=texture(T,v).r; else if(ch<1.5) c=texture(T,v).g; else if(ch<2.5) c=texture(T,v).b; else c=texture(T,v).a; o=vec4(c,c,c,1.0); }`;
const FS_COLOR = `#version 300 es\nprecision highp float; uniform sampler2D T; uniform float brightness, contrast, saturation; in vec2 v; out vec4 o; void main(){ vec3 c=texture(T,v).rgb; c=(c-0.5)*contrast+0.5; float lum=dot(c,vec3(0.2126,0.7152,0.0722)); c=mix(vec3(lum),c,saturation)*brightness; o=vec4(clamp(c,0.0,1.0),1.0); }`;
const FS_SEAMLESS = `#version 300 es\nprecision highp float; uniform sampler2D T; in vec2 v; out vec4 o; ${GLSL_COMMON} void main(){ vec2 uv=mod(v,1.0); vec2 uv2=mod(v+0.5,1.0); float n=fbm(v*15.0)*0.15-0.075; float wx=smoothstep(0.2,0.8,abs(v.x-0.5)*2.0+n), wy=smoothstep(0.2,0.8,abs(v.y-0.5)*2.0+n); o=mix(texture(T,uv),texture(T,uv2),max(wx,wy)); }`;
const FS_BH = `#version 300 es\nprecision highp float;\nuniform sampler2D T;uniform vec2 px;uniform float radius;in vec2 v;out vec4 o;const float W[5]=float[5](0.227027,0.194595,0.121622,0.054054,0.016216);void main(){vec4 c=texture(T,v)*W[0];for(int i=1;i<5;i++)c+=texture(T,v+vec2(float(i)*px.x*radius,0.))*W[i]+texture(T,v-vec2(float(i)*px.x*radius,0.))*W[i];o=c;}`;
const FS_BV = `#version 300 es\nprecision highp float;\nuniform sampler2D T;uniform vec2 px;uniform float radius;in vec2 v;out vec4 o;const float W[5]=float[5](0.227027,0.194595,0.121622,0.054054,0.016216);void main(){vec4 c=texture(T,v)*W[0];for(int i=1;i<5;i++)c+=texture(T,v+vec2(0.,float(i)*px.y*radius))*W[i]+texture(T,v-vec2(0.,float(i)*px.y*radius))*W[i];o=c;}`;
const FS_HP = `#version 300 es\nprecision highp float;\nuniform sampler2D ORIG,BLUR;in vec2 v;out vec4 o;void main(){o=clamp(texture(ORIG,v)-texture(BLUR,v)+vec4(.5),0.,1.);}`;

// 🔥 [분위기 융합 적용] 칼같은 경계를 허물고 전역 베이스(35%)와 특정영역 타겟(100%)을 블렌딩
const FS_PROC_HEIGHT = `#version 300 es\nprecision highp float; uniform sampler2D SEG; uniform float rustLvl, scratchLvl, microType, microScale, microDepth, uTile; in vec2 v; out vec4 o; ${GLSL_COMMON} void main(){ 
  vec2 uv = mod(v * uTile, 1.0); 
  float rust = 1.0 - voronoi(uv * 15.0); rust = smoothstep(0.4, 0.8, rust + (fbm(uv * 5.0) * 0.5)) * rustLvl; 
  float scr = max(scratch(uv, 0.5, 50.0), scratch(uv, -0.2, 40.0)) * scratchLvl; 
  float micro = 0.0; vec2 muv = uv * microScale; float weight = 1.0; 
  
  if (microType > 0.0) {
    float seg_val = texture(SEG, v).r;
    int cls = int(round(seg_val * 4.0));
    
    if (microType == 1.0) { micro = hash(muv); weight = step(1.5, float(cls)); }
    else if (microType == 2.0) { micro = pow(1.0 - voronoi(muv), 3.0); weight = step(2.5, float(cls)) * step(float(cls), 3.5); }
    else if (microType == 3.0) { micro = (sin(muv.x) * sin(muv.y)) * 0.5 + 0.5; weight = step(1.5, float(cls)) * step(float(cls), 3.5); }
    else if (microType == 4.0) { micro = noise(vec2(muv.x * 0.05, muv.y)); weight = step(0.5, float(cls)) * step(float(cls), 1.5) + step(3.5, float(cls)); }
    else if (microType == 5.0) { micro = smoothstep(0.2, 0.8, voronoi(muv * 0.5)); weight = step(0.5, float(cls)) * step(float(cls), 2.5); }
    else if (microType == 6.0) { micro = pow(abs(noise(muv) - 0.5) * 2.0, 4.0); weight = step(1.5, float(cls)) * step(float(cls), 3.5); }
  }
  
  // 🔥 핵심 로직: 0%로 잘라내지 않고, 기본값 0.35를 주어 화면 전체 분위기를 묶음
  float micro_apply_mask = mix(0.35, 1.0, weight); 
  micro *= microDepth * micro_apply_mask; 
  o = vec4(rust, scr, micro, 1.0); 
}`;

const FS_NM = `#version 300 es
precision highp float; uniform sampler2D DEPTH,PROC; uniform vec2 px; uniform float str,microDetail,invertY; in vec2 v;out vec4 o; ${GLSL_COMMON}
float hD(vec2 u){return texture(DEPTH,u).r;}
void main(){ 
  float dc=hD(v); float dx=0.,dy=0.;
  vec2 ofs[8] = vec2[](px*vec2(-1,1), px*vec2(0,1), px*vec2(1,1), px*vec2(-1,0), px*vec2(1,0), px*vec2(-1,-1), px*vec2(0,-1), px*vec2(1,-1));
  float kx[8] = float[](-1.,0.,1.,-2.,2.,-1.,0.,1.); float ky[8] = float[](1.,2.,1.,0.,0.,-1.,-2.,-1.);
  for(int i=0;i<8;i++){ float ds=hD(v+ofs[i]); dx+=kx[i]*ds; dy+=ky[i]*ds; } 
  vec3 nBase = normalize(vec3(dx*str, dy*str, 1.0));
  vec4 p = texture(PROC, v); float detailH = (p.r + p.g)*0.5 + p.b; 
  float ddx = (texture(PROC, v + vec2(px.x, 0.)).b - texture(PROC, v - vec2(px.x, 0.)).b) * 2.0;
  float ddy = (texture(PROC, v + vec2(0., px.y)).b - texture(PROC, v - vec2(0., px.y)).b) * 2.0;
  vec3 nDetail = normalize(vec3(-ddx * microDetail * 20.0, -ddy * microDetail * 20.0, 1.0));
  vec3 t = nBase + vec3(0., 0., 1.); vec3 u = nDetail * vec3(-1., -1., 1.);
  vec3 n = normalize(t * dot(t, u) - u * t.z);
  if(invertY>0.5){n.y=-n.y;} o=vec4(n*.5+.5,1.); 
}`;

const FS_CURV = `#version 300 es\nprecision highp float;\nuniform sampler2D NM;uniform vec2 px;uniform float scale;in vec2 v;out vec4 o;float cs(float s){vec3 n=normalize(texture(NM,v).rgb*2.-1.);float d=0.;vec2 ds[4];ds[0]=vec2(px.x*s,0.);ds[1]=vec2(0.,px.y*s);ds[2]=vec2(-px.x*s,0.);ds[3]=vec2(0.,-px.y*s);for(int i=0;i<4;i++)d+=dot(n,normalize(texture(NM,v+ds[i]).rgb*2.-1.));return clamp(-(d/4.-1.)*scale*3.+.5,0.,1.);}void main(){float c=cs(1.)*.5+cs(4.)*.3+cs(12.)*.2;o=vec4(c,c,c,1.);}`;

const FS_MASKS = `#version 300 es
precision highp float; uniform sampler2D CURV; uniform float edgeWear, edgeContrast, grungeLvl; in vec2 v; out vec4 o; ${GLSL_COMMON}
void main() { 
  float curv = texture(CURV, v).r; 
  float cavity = clamp((0.5 - curv) * 4.0, 0.0, 1.0); 
  float edge = clamp((curv - 0.5) * 4.0, 0.0, 1.0);   
  float n = fbm(v * vec2(8.0, 12.0)); 
  float grunge = clamp((n - 0.5 + grungeLvl) * 2.5, 0.0, 1.0); 
  float wearMask = clamp((edge - (1.0 - edgeWear)) * edgeContrast, 0.0, 1.0) * grunge;
  o = vec4(cavity, wearMask, grunge, 1.0); 
}`;

const FS_ROUGH = `#version 300 es
precision highp float; uniform sampler2D DEPTH,DETAIL,SEG,MASKS,PROC; 
uniform vec2 px; uniform float con,bias,inv,detailMix,matBlend,metalMode,roughMin,roughMax; 
uniform float uRough[5]; in vec2 v; out vec4 o; 
void main(){ 
  float sd=0.,sd2=0.; for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++){float s=texture(DEPTH,v+px*vec2(float(dx),float(dy))).r;sd+=s;sd2+=s*s;} sd/=25.;sd2/=25.;float lv=max(sd2-sd*sd,0.)*80.; 
  float fe=length(texture(DETAIL,v).rgb-vec3(.5))*2.; float lb=texture(DEPTH,v).r;float l=mix(lb,mix(lv,fe,.5),detailMix); 
  int cls = clamp(int(round(texture(SEG,v).r * 4.0)), 0, 4); float rp = uRough[cls]; 
  l=mix(l,rp,matBlend); vec4 m = texture(MASKS, v); l = clamp(l + m.r * 0.5, 0.0, 1.0); 
  if(metalMode > 0.5) { l = mix(l, 0.95, m.g); } 
  l = clamp(l + texture(PROC, v).b * 0.8, 0.0, 1.0); l=mix(l,1.-l,inv); 
  float finalR = clamp((l-.5)*con+.5+bias, 0.0, 1.0);
  o=vec4(vec3(clamp(finalR, roughMin, roughMax)),1.); 
}`;

const FS_METAL = `#version 300 es\nprecision highp float; uniform sampler2D T,SEG,MASKS; uniform float thr,soft,metalMode; in vec2 v; out vec4 o; void main(){ float l=texture(T,v).r; float cls = round((texture(SEG,v).r) * 4.0); float bias=0.; if(cls>0.5&&cls<1.5)bias=-0.28; else if(cls>3.5)bias=0.18; float mt=smoothstep(thr+bias-soft,thr+bias+soft,l); vec4 m = texture(MASKS, v); if(metalMode > 0.5) { mt = mix(mt, 0.0, m.r); mt = mix(mt, 1.0, m.g); } else { mt = clamp(mt + m.g * 0.8, 0.0, 1.0); } o=vec4(mt,mt,mt,1.); }`;
const FS_DISP = `#version 300 es\nprecision highp float;\nuniform sampler2D T;uniform float mid,con,inv;in vec2 v;out vec4 o;void main(){float d=texture(T,v).r;d=clamp((d-mid)*con+.5,0.,1.);d=mix(d,1.-d,inv);o=vec4(d,d,d,1.);}`;
const FS_AO = `#version 300 es\nprecision highp float;\nuniform sampler2D T, MASKS; uniform vec2 px; uniform float radius,strength; in vec2 v; out vec4 o; const float G=2.3999632; const int S=32; void main(){ float dc=texture(T,v).r,occ=0.,dMn=1.,dMx=0.; for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++){float s=texture(T,v+px*vec2(float(dx),float(dy))*4.).r;dMn=min(dMn,s);dMx=max(dMx,s);} float dr=max(dMx-dMn,.01),ar=radius/dr; for(int i=0;i<S;i++){float r=sqrt(float(i+1)/float(S)),th=float(i)*G; vec2 off=vec2(cos(th),sin(th))*px*min(ar,radius)*80.*r; float diff=clamp((dc-texture(T,v+off).r)*8.,0.,1.); float cosW=(1.-r)*.7+.3; occ+=diff*cosW; } float ao=clamp(1.-(occ/float(S)*2.)*strength,0.,1.); float cavity = texture(MASKS, v).r; ao = clamp(ao - cavity * 0.5, 0.0, 1.0); o=vec4(ao,ao,ao,1.); }`;
const FS_ORM = `#version 300 es\nprecision highp float;\nuniform sampler2D AO,RO,ME;in vec2 v;out vec4 o;void main(){o=vec4(texture(AO,v).r,texture(RO,v).r,texture(ME,v).r,1.);}`;
const FS_ALBEDO_FINAL = `#version 300 es\nprecision highp float; uniform sampler2D T, MASKS, PROC; uniform float metalMode; in vec2 v; out vec4 o; void main(){ vec4 c = texture(T, v); float micro = texture(PROC, v).b; c.rgb *= (1.0 - micro * 0.4); if(metalMode > 0.5) { vec4 m = texture(MASKS, v); float lum = dot(c.rgb, vec3(0.2126,0.7152,0.0722)); vec3 rColor = vec3(0.35, 0.15, 0.05) * (0.8 + 0.4*lum); c.rgb = mix(c.rgb, rColor, m.r); c.rgb = mix(c.rgb, c.rgb + 0.3, m.g); } o = vec4(clamp(c.rgb, 0.0, 1.0), c.a); }`;
const FS_PHYSICAL_BAKE = `#version 300 es\nprecision highp float; uniform sampler2D SEG; uniform float uCC[5], uSheen[5], uTrans[5]; in vec2 v; out vec4 o; void main() { int cls = clamp(int(round(texture(SEG,v).r * 4.0)), 0, 4); o = vec4(uCC[cls], uSheen[cls], uTrans[cls], 1.0); }`;

/* ═══════════════════════════════════════════════════════════════
   4. WEBGL2 PIPELINE & MAP GENERATION
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
    const shaders = [];
    for (const [t, src] of [
      [gl.VERTEX_SHADER, VS],
      [gl.FRAGMENT_SHADER, fs],
    ]) {
      const sh = gl.createShader(t);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      gl.attachShader(p, sh);
      shaders.push(sh);
    }
    gl.linkProgram(p);
    shaders.forEach((sh) => {
      gl.detachShader(p, sh);
      gl.deleteShader(sh);
    });
    return p;
  };
  const P = {
    pass: mp(FS_PASS),
    channel: mp(FS_CHANNEL),
    color: mp(FS_COLOR),
    albedoFinal: mp(FS_ALBEDO_FINAL),
    hqSeam: mp(FS_SEAMLESS),
    bh: mp(FS_BH),
    bv: mp(FS_BV),
    hp: mp(FS_HP),
    procHeight: mp(FS_PROC_HEIGHT),
    nm: mp(FS_NM),
    curv: mp(FS_CURV),
    masks: mp(FS_MASKS),
    disp: mp(FS_DISP),
    rough: mp(FS_ROUGH),
    metal: mp(FS_METAL),
    ao: mp(FS_AO),
    orm: mp(FS_ORM),
    physBake: mp(FS_PHYSICAL_BAKE),
  };
  const qb = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, qb);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
    gl.STATIC_DRAW,
  );
  const mkt = () => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
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
  const T = {
    origA: mkt(),
    a: mkt(),
    d: mkt(),
    s: mkt(),
    sa: mkt(),
    sd: mkt(),
    bh: mkt(),
    bv: mkt(),
    hi: mkt(),
    proc: mkt(),
    nm: mkt(),
    cv: mkt(),
    masks: mkt(),
    ao: mkt(),
    ro: mkt(),
    me: mkt(),
    di: mkt(),
    finalA: mkt(),
    phys: mkt(),
  };
  const F = {};
  for (const k of Object.keys(T)) {
    const f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      T[k],
      0,
    );
    F[k] = f;
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

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
      if (u !== null) {
        if (Array.isArray(v)) {
          if (v.length === 2) gl.uniform2fv(u, v);
          else gl.uniform1fv(u, new Float32Array(v));
        } else gl.uniform1f(u, v);
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo ?? null);
    gl.viewport(0, 0, W, H);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  };
  const upt = (t, s) => {
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, s);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  };
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
  const { c, P, T, F, exec, upt } = ctx;
  const px = [1 / W, 1 / H];
  upt(T.origA, albC);
  upt(T.d, depC);
  upt(T.s, segC);

  exec(P.color, F.a, [["T", T.origA]], {
    brightness: s.brightness,
    contrast: s.albContrast,
    saturation: s.saturation,
  });
  if (s.seamless) {
    exec(P.hqSeam, F.sa, [["T", T.a]], {});
    exec(P.hqSeam, F.sd, [["T", T.d]], {});
  }
  const srcA = s.seamless ? T.sa : T.a;
  const srcD = s.seamless ? T.sd : T.d;

  exec(P.bh, F.bh, [["T", srcA]], { px, radius: s.blurRadius });
  exec(P.bv, F.bv, [["T", T.bh]], { px, radius: s.blurRadius });
  exec(
    P.hp,
    F.hi,
    [
      ["ORIG", srcA],
      ["BLUR", T.bv],
    ],
    {},
  );

  // 🔥 [UX+질감 융합] ProcHeight에 SEG 바인딩 추가
  exec(P.procHeight, F.proc, [["SEG", T.s]], {
    rustLvl: s.metalMode ? s.rustLvl : 0,
    scratchLvl: s.metalMode ? s.scratchLvl : 0,
    microType: s.microType,
    microScale: s.microScale,
    microDepth: s.microDepth,
    uTile: s.tileRep,
  });

  exec(
    P.nm,
    F.nm,
    [
      ["DEPTH", srcD],
      ["PROC", T.proc],
    ],
    {
      px,
      str: s.normalStr,
      microDetail: s.microDetail,
      invertY: s.normalInvertY ? 1 : 0,
    },
  );
  exec(P.curv, F.cv, [["NM", T.nm]], { px, scale: s.curvScale });
  exec(
    P.masks,
    F.masks,
    [
      ["CURV", T.cv],
      ["PROC", T.proc],
    ],
    {
      edgeWear: s.edgeWear,
      edgeContrast: s.edgeContrast,
      grungeLvl: s.grungeLvl,
    },
  );

  exec(P.physBake, F.phys, [["SEG", T.s]], {
    uCC: MAT_PROFILES.cc,
    uSheen: MAT_PROFILES.sheen,
    uTrans: MAT_PROFILES.trans,
  });
  exec(
    P.rough,
    F.ro,
    [
      ["DEPTH", srcD],
      ["DETAIL", T.hi],
      ["SEG", T.s],
      ["MASKS", T.masks],
      ["PROC", T.proc],
    ],
    {
      px,
      con: s.roughCon,
      bias: s.roughBias,
      inv: s.roughInv ? 1 : 0,
      detailMix: s.roughDetailMix,
      matBlend: s.matBlend,
      metalMode: s.metalMode ? 1 : 0,
      roughMin: s.roughMin,
      roughMax: s.roughMax,
      uRough: MAT_PROFILES.rough,
    },
  );

  exec(
    P.metal,
    F.me,
    [
      ["T", srcD],
      ["SEG", T.s],
      ["MASKS", T.masks],
    ],
    { thr: s.metalThr, soft: s.metalSoft, metalMode: s.metalMode ? 1 : 0 },
  );
  exec(
    P.ao,
    F.ao,
    [
      ["T", srcD],
      ["MASKS", T.masks],
    ],
    { px, radius: s.aoRadius, strength: s.aoStr },
  );
  exec(P.disp, F.di, [["T", srcD]], {
    mid: s.dispMid,
    con: s.dispCon,
    inv: s.dispInv ? 1 : 0,
  });
  exec(
    P.albedoFinal,
    F.finalA,
    [
      ["T", srcA],
      ["MASKS", T.masks],
      ["PROC", T.proc],
    ],
    { metalMode: s.metalMode ? 1 : 0 },
  );

  const snap = (prog, binds, uns = {}) => {
    exec(prog, null, binds, uns);
    const res = document.createElement("canvas");
    res.width = W;
    res.height = H;
    res.getContext("2d").drawImage(c, 0, 0);
    return res;
  };

  return {
    albedo: snap(P.pass, [["T", T.finalA]]),
    normal: snap(P.pass, [["T", T.nm]]),
    displacement: snap(P.pass, [["T", T.di]]),
    roughness: snap(P.pass, [["T", T.ro]]),
    metallic: snap(P.pass, [["T", T.me]]),
    ao: snap(P.pass, [["T", T.ao]]),
    orm: snap(P.orm, [
      ["AO", T.ao],
      ["RO", T.ro],
      ["ME", T.me],
    ]),
    ccMask: snap(P.channel, [["T", T.phys]], { ch: 0 }),
    sheenMask: snap(P.channel, [["T", T.phys]], { ch: 1 }),
    transMask: snap(P.channel, [["T", T.phys]], { ch: 2 }),
    cavityMap: snap(P.channel, [["T", T.masks]], { ch: 0 }),
    edgeWearMap: snap(P.channel, [["T", T.masks]], { ch: 1 }),
  };
}

async function generateMaps(img, depCvs, segCvs, s, res, onProg) {
  const sc = res / Math.max(img.naturalWidth, img.naturalHeight);
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

  let resultCanvases;
  if (W <= TILE && H <= TILE) {
    const ctx = mkGL(W, H);
    resultCanvases = runTile(
      ctx,
      rgba(img, W, H),
      rgba(depCvs, W, H),
      rgba(neutral, W, H),
      W,
      H,
      s,
    );
  } else {
    const TW = TILE + 2 * PAD,
      TH = TILE + 2 * PAD;
    const ctx = mkGL(TW, TH);
    const txA = Math.ceil(W / TILE),
      tyA = Math.ceil(H / TILE),
      tot = txA * tyA;
    const fA = rgba(img, W, H),
      fD = rgba(depCvs, W, H),
      fS = rgba(neutral, W, H);
    const keys = [
      "albedo",
      "normal",
      "displacement",
      "roughness",
      "metallic",
      "ao",
      "orm",
      "ccMask",
      "sheenMask",
      "transMask",
      "cavityMap",
      "edgeWearMap",
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
          [tA, fA],
          [tD, fD],
          [tS, fS],
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
          if (!r[k]) continue;
          outs[k]
            .getContext("2d")
            .drawImage(r[k], PAD, PAD, cW, cH, dX, dY, cW, cH);
          r[k].width = 0;
          r[k].height = 0;
        }
        tn++;
        onProg?.(tn, tot);
        await new Promise((r) => setTimeout(r, 0));
      }
    resultCanvases = outs;
  }

  const out = {};
  for (const [k, cvs] of Object.entries(resultCanvases)) {
    if (!cvs || cvs.width === 0) continue;
    const blob = await new Promise((resolve) =>
      cvs.toBlob(resolve, "image/png"),
    );
    if (blob) out[k] = URL.createObjectURL(blob);
    cvs.width = 0;
    cvs.height = 0;
  }
  return out;
}

/* ═══════════════════════════════════════════════════════════════
   5. UI CONSTANTS & APP COMPONENT
═══════════════════════════════════════════════════════════════ */
const DEF = {
  metalMode: false,
  seamless: false,
  brightness: 1,
  albContrast: 1,
  saturation: 1,
  microType: 0,
  microScale: 400,
  microDepth: 0.5,
  edgeWear: 0.4,
  edgeContrast: 2.5,
  grungeLvl: 0.3,
  microDetail: 0.4,
  rustLvl: 0.5,
  scratchLvl: 0.5,
  normalStr: 3,
  detailMix: 0.4,
  normalSigma: 0.05,
  normalInvertY: false,
  dispMid: 0.5,
  dispCon: 2,
  dispInv: false,
  dispScale: 0.05,
  roughCon: 2,
  roughBias: 0,
  roughMin: 0.0,
  roughMax: 1.0,
  roughInv: false,
  roughDetailMix: 0.5,
  matBlend: 0.25,
  metalThr: 0.75,
  metalSoft: 0.1,
  blurRadius: 6,
  aoRadius: 0.8,
  aoStr: 1.2,
  curvScale: 1,
  envRotation: 0,
  envExposure: 1.0,
  tileRep: 1,
  shape: "sphere",
  hdri: "studio",
  res: 1024,
  quality: false,
};
const C = {
  bg: "#0e0f14",
  sidebar: "#13141a",
  panel: "#1a1b22",
  border: "#2a2b38",
  text: "#c8cad8",
  textDim: "#6a6c80",
  accent: "#e8a020",
  blue: "#5090e8",
  green: "#40c060",
  purple: "#9060d0",
  teal: "#30b898",
  red: "#c04030",
};
const MAP_DISPLAY = [
  { k: "albedo", l: "Color", c: C.green },
  { k: "normal", l: "Normal", c: C.blue },
  { k: "displacement", l: "Depth", c: C.accent },
  { k: "roughness", l: "Rough", c: C.purple },
  { k: "metallic", l: "Metal", c: C.teal },
  { k: "ao", l: "AO", c: C.red },
  { k: "cavityMap", l: "Cavity", c: C.accent },
  { k: "edgeWearMap", l: "Edge Mask", c: C.green },
  { k: "ccMask", l: "Clearcoat", c: "#fff" },
  { k: "sheenMask", l: "Sheen", c: "#f0f" },
  { k: "transMask", l: "Trans", c: "#0ff" },
];
const S = {
  idle: { l: "Idle", d: C.textDim },
  loading: { l: "AI Loading…", d: C.blue },
  inferring: { l: "AI Inferring…", d: C.green },
  segload: { l: "Seg Loading…", d: C.purple },
  seginfer: { l: "Analyzing…", d: C.purple },
  segmask: { l: "Masking…", d: C.purple },
  generating: { l: "Generating…", d: C.accent },
  tiling: { l: "Tiling…", d: C.accent },
  exporting: { l: "Baking GLB…", d: C.accent },
  ready: { l: "Ready", d: C.green },
  error: { l: "Error", d: C.red },
};

export default function App() {
  const [srcURL, setSrcURL] = useState(null);
  const [maps, setMaps] = useState(null);
  const [settings, setSettings] = useState(DEF);
  const [proc, setProc] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [imgInfo, setImgInfo] = useState(null);
  const [aiStep, setAiStep] = useState("idle");
  const [tileProg, setTileProg] = useState(null);
  const [gpu, setGpu] = useState({ maxTex: 4096 });
  const cvs = useRef(null),
    imgR = useRef(null),
    depR = useRef(null),
    segR = useRef(null),
    threeR = useRef({}),
    tok = useRef(0);

  useEffect(() => {
    setGpu(queryGPU());
  }, []);

  const generate = useCallback(async (img, dep, seg, s, r) => {
    const tid = ++tok.current;
    setProc(true);
    setTileProg(null);
    setAiStep(r > TILE ? "tiling" : "generating");
    try {
      const safeSrc = getSafeImage(img, 1024);
      if (s.quality && !segR.current) {
        setAiStep("segload");
        const sg = await getSeg(true);
        if (tid !== tok.current) return;
        setAiStep("seginfer");
        const sr = await sg(safeSrc);
        if (tid !== tok.current) return;
        setAiStep("segmask");
        segR.current = await buildMatMask(sr);
      }
      const res = await generateMaps(
        img,
        dep,
        s.quality ? segR.current : null,
        s,
        r,
        (c, t) => {
          if (tid === tok.current) setTileProg({ c, t });
        },
      );

      if (tid !== tok.current) {
        Object.values(res).forEach((u) => u && URL.revokeObjectURL(u));
        return;
      }

      setMaps((prev) => {
        if (prev)
          Object.values(prev).forEach((u) => u && URL.revokeObjectURL(u));
        return res;
      });
      setAiStep("ready");
    } catch (e) {
      if (tid !== tok.current) return;
      console.error(e);
      setAiStep("error");
    } finally {
      if (tid === tok.current) {
        setTileProg(null);
        setProc(false);
      }
    }
  }, []);

  const loadFile = (file) => {
    if (!file?.type.startsWith("image/")) return;
    const tid = ++tok.current,
      url = URL.createObjectURL(file),
      img = new Image();
    img.onload = async () => {
      if (tid !== tok.current) return;
      imgR.current = img;
      setSrcURL(url);
      setImgInfo({ name: file.name.split(".")[0] });

      setMaps((prev) => {
        if (prev)
          Object.values(prev).forEach((u) => u && URL.revokeObjectURL(u));
        return null;
      });

      depR.current = segR.current = null;
      setProc(true);
      try {
        const safeSrc = getSafeImage(imgR.current, 1024);
        setAiStep("loading");
        const dp = await getDepth();
        if (tid !== tok.current) return;
        setAiStep("inferring");
        const dr = await dp(safeSrc);
        if (tid !== tok.current) return;
        depR.current = dr.depth.toCanvas();
        if (settings.quality) {
          setAiStep("segload");
          const sg = await getSeg(true);
          if (tid !== tok.current) return;
          setAiStep("seginfer");
          const sr = await sg(safeSrc);
          if (tid !== tok.current) return;
          setAiStep("segmask");
          segR.current = await buildMatMask(sr);
        }
        await generate(
          imgR.current,
          depR.current,
          segR.current,
          settings,
          settings.res,
        );
      } catch (e) {
        if (tid !== tok.current) return;
        setAiStep("error");
        setProc(false);
      }
    };
    img.src = url;
  };

  useEffect(() => {
    if (!imgR.current || !depR.current) return;
    const id = setTimeout(
      () =>
        generate(
          imgR.current,
          depR.current,
          segR.current,
          settings,
          settings.res,
        ),
      400,
    );
    return () => clearTimeout(id);
  }, [settings, generate]);

  useEffect(() => {
    const el = cvs.current;
    const renderer = new THREE.WebGLRenderer({
      canvas: el,
      antialias: true,
      preserveDrawingBuffer: true,
    });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      45,
      el.clientWidth / el.clientHeight,
      0.1,
      100,
    );
    camera.position.set(0, 0, 3);
    const env = buildEnv(renderer, "studio");
    scene.environment = scene.background = env;
    const light = new THREE.DirectionalLight(0xfff4d6, 0.6);
    light.position.set(4, 5, 3);
    scene.add(light);
    const G = {
      sphere: new THREE.SphereGeometry(1, 256, 128),
      plane: new THREE.PlaneGeometry(2, 2, 256, 256),
      torus: new THREE.TorusGeometry(0.7, 0.3, 128, 256),
    };
    Object.values(G).forEach((g) => g.setAttribute("uv2", g.attributes.uv));
    const mat = new THREE.MeshPhysicalMaterial({
      roughness: 1,
      metalness: 0,
      color: 0xd0d0d0,
      ior: 1.5,
      thickness: 0.5,
    });
    const mesh = new THREE.Mesh(G.sphere, mat);
    scene.add(mesh);
    threeR.current = { renderer, scene, camera, mat, mesh, G, light };
    const resize = () => {
      renderer.setSize(el.clientWidth, el.clientHeight, false);
      camera.aspect = el.clientWidth / el.clientHeight;
      camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", resize);
    resize();

    let raf;
    const orbit = { on: false, x: 0, y: 0 };
    const tick = () => {
      raf = requestAnimationFrame(tick);
      renderer.render(scene, camera);
    };
    tick();
    el.onmousedown = (e) => {
      orbit.on = true;
      orbit.x = e.clientX;
      orbit.y = e.clientY;
    };
    window.onmouseup = () => (orbit.on = false);
    window.onmousemove = (e) => {
      if (!orbit.on) return;
      mesh.rotation.y += (e.clientX - orbit.x) * 0.01;
      mesh.rotation.x += (e.clientY - orbit.y) * 0.01;
      orbit.x = e.clientX;
      orbit.y = e.clientY;
    };

    // 🔥 [V9.7 UX] 마우스 휠 스크롤 줌인/아웃 구현
    const handleWheel = (e) => {
      e.preventDefault();
      camera.position.z += e.deltaY * 0.005;
      camera.position.z = Math.max(1.2, Math.min(camera.position.z, 8.0));
    };
    el.addEventListener("wheel", handleWheel, { passive: false });

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      el.removeEventListener("wheel", handleWheel);
      renderer.dispose();
    };
  }, []);

  useEffect(() => {
    if (threeR.current.renderer) {
      threeR.current.renderer.toneMappingExposure = settings.envExposure;
    }
  }, [settings.envExposure]);

  useEffect(() => {
    const { renderer, scene } = threeR.current;
    if (renderer) {
      const e = buildEnv(renderer, settings.hdri);
      scene.environment = scene.background = e;
    }
  }, [settings.hdri]);
  useEffect(() => {
    const { mesh, G } = threeR.current;
    if (mesh) mesh.geometry = G[settings.shape];
  }, [settings.shape]);

  const buildUnifiedMaterial = async (mapData, tileRep, isExport = false) => {
    const loader = new THREE.TextureLoader();
    const loadTex = (url, isColor = false) =>
      new Promise((resolve) => {
        if (!url) return resolve(null);
        loader.load(
          url,
          (t) => {
            t.wrapS = t.wrapT = THREE.RepeatWrapping;
            t.repeat.set(tileRep, tileRep);
            t.flipY = !isExport;
            if (isColor) t.colorSpace = THREE.SRGBColorSpace;
            resolve(t);
          },
          undefined,
          () => resolve(null),
        );
      });
    const [alb, nrm, orm, cc, sh, tr, disp, cav, edge] = await Promise.all([
      loadTex(mapData.albedo, true),
      loadTex(mapData.normal, false),
      loadTex(mapData.orm, false),
      loadTex(mapData.ccMask, false),
      loadTex(mapData.sheenMask, false),
      loadTex(mapData.transMask, false),
      loadTex(mapData.displacement, false),
      loadTex(mapData.cavityMap, false),
      loadTex(mapData.edgeWearMap, false),
    ]);
    const m = new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      roughness: 1,
      metalness: orm ? 1 : 0,
    });
    if (alb) m.map = alb;
    if (nrm) {
      m.normalMap = nrm;
      m.clearcoatNormalMap = nrm;
    }
    if (orm) {
      m.roughnessMap = m.metalnessMap = orm;
    }
    if (cav) {
      m.aoMap = cav;
      m.aoMapIntensity = 1;
      m.roughness = Math.min(1, m.roughness + 0.12);
    } else if (orm) {
      m.aoMap = orm;
      m.aoMapIntensity = 1;
    }
    if (disp && !isExport) m.displacementMap = disp;
    if (edge) {
      m.clearcoatMap = edge;
      m.clearcoat = 1;
      m.clearcoatRoughness = 0.2;
    } else if (cc) {
      m.clearcoatMap = cc;
      m.clearcoat = 1;
      m.clearcoatRoughness = 0.1;
    }
    if (sh) {
      m.sheenColorMap = sh;
      m.sheen = 1;
      m.sheenRoughness = 0.6;
    }
    if (tr) {
      m.transmissionMap = tr;
      m.transmission = 1;
      m.ior = 1.5;
      m.thickness = 0.5;
    }
    return m;
  };

  useEffect(() => {
    if (!maps || !threeR.current?.mesh) return;
    buildUnifiedMaterial(maps, settings.tileRep, false).then((mat) => {
      mat.displacementScale = settings.dispScale;
      if (threeR.current.mat) threeR.current.mat.dispose();
      threeR.current.mat = mat;
      threeR.current.mesh.material = mat;
      threeR.current.mat.needsUpdate = true;
    });
  }, [maps, settings.tileRep, settings.dispScale]);

  useEffect(() => {
    const { scene, light } = threeR.current;
    if (light) {
      scene.environmentRotation.y = settings.envRotation;
      light.position.set(
        Math.cos(settings.envRotation) * 5,
        5,
        Math.sin(settings.envRotation) * 5,
      );
    }
  }, [settings.envRotation]);

  const ss = (k) => (e) =>
    setSettings((s) => ({ ...s, [k]: parseFloat(e.target.value) }));
  const tg = (k) => () => setSettings((s) => ({ ...s, [k]: !s[k] }));
  // 🔥 [V9.7 UX] 더블클릭 초기화 연결용 핸들러
  const sr = (k) => () => setSettings((s) => ({ ...s, [k]: DEF[k] }));

  const handleGLB = async () => {
    if (!maps || exporting || !threeR.current?.mesh) return;
    setExporting(true);
    setAiStep("exporting");
    let objectUrl = null;
    const { mesh } = threeR.current;
    const exportMesh = mesh.clone(false);
    const exportGeom = mesh.geometry.clone();
    try {
      if (exportGeom.attributes.uv && !exportGeom.attributes.uv2)
        exportGeom.setAttribute(
          "uv2",
          new THREE.BufferAttribute(
            new Float32Array(exportGeom.attributes.uv.array),
            2,
          ),
        );
      if (
        maps.displacement &&
        exportGeom.attributes.position &&
        exportGeom.attributes.normal &&
        exportGeom.attributes.uv
      ) {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.src = maps.displacement;
        await new Promise((res, rej) => {
          img.onload = res;
          img.onerror = () => rej(new Error("Disp load fail"));
        });
        const cvs = document.createElement("canvas");
        cvs.width = img.width;
        cvs.height = img.height;
        const ctx = cvs.getContext("2d", { willReadFrequently: true });
        ctx.drawImage(img, 0, 0);
        const imgData = ctx.getImageData(0, 0, img.width, img.height).data;
        const pos = exportGeom.attributes.position,
          norm = exportGeom.attributes.normal,
          uv = exportGeom.attributes.uv;
        for (let i = 0; i < pos.count; i++) {
          const u = (((uv.getX(i) * settings.tileRep) % 1) + 1) % 1,
            v = 1.0 - ((((uv.getY(i) * settings.tileRep) % 1) + 1) % 1);
          const tx = Math.min(img.width - 1, Math.floor(u * (img.width - 1))),
            ty = Math.min(img.height - 1, Math.floor(v * (img.height - 1)));
          const dispValue =
            (0.299 * imgData[(ty * img.width + tx) * 4] +
              0.587 * imgData[(ty * img.width + tx) * 4 + 1] +
              0.114 * imgData[(ty * img.width + tx) * 4 + 2]) /
            255.0;
          const disp = (dispValue - settings.dispMid) * settings.dispScale;
          pos.setXYZ(
            i,
            pos.getX(i) + norm.getX(i) * disp,
            pos.getY(i) + norm.getY(i) * disp,
            pos.getZ(i) + norm.getZ(i) * disp,
          );
        }
        pos.needsUpdate = true;
        exportGeom.computeVertexNormals();
      }
      const exportMat = await buildUnifiedMaterial(
        maps,
        settings.tileRep,
        true,
      );
      exportMesh.geometry = exportGeom;
      exportMesh.material = exportMat;
      exportMesh.updateMatrixWorld(true);
      const glb = await new Promise((resolve, reject) => {
        new GLTFExporter().parse(exportMesh, resolve, reject, {
          binary: true,
          embedImages: true,
          forceIndices: true,
          includeCustomExtensions: true,
        });
      });
      const blob = new Blob([glb], { type: "application/octet-stream" });
      objectUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objectUrl;
      a.download = `Physical_${imgInfo?.name || "PBR"}.glb`;
      a.click();
      setTimeout(() => {
        if (objectUrl) URL.revokeObjectURL(objectUrl);
      }, 1000);
      setAiStep("ready");
    } catch (e) {
      console.error(e);
      alert("GLB Export Failed.");
      setAiStep("error");
    } finally {
      if (exportMesh.geometry !== mesh.geometry) exportMesh.geometry.dispose();
      exportMesh.material.dispose();
      setExporting(false);
    }
  };

  const pName = () => imgInfo?.name || "PBR";
  const handleZip = async () => {
    if (!maps || exporting) return;
    setExporting(true);
    const z = new JSZip(),
      f = z.folder(pName());
    await Promise.all(
      Object.entries(maps).map(async ([k, u]) => {
        if (u) f.file(`${pName()}_${k}.png`, await (await fetch(u)).blob());
      }),
    );
    const b = await z.generateAsync({ type: "blob" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    a.download = `${pName()}.zip`;
    a.click();
    setExporting(false);
  };

  const handleMTLX = () => {
    if (!maps) return;
    const p = pName();
    const f = (suffix) => `${p}_${suffix}.png`;
    let nodesXML = `
      <image name="tex_color" type="color3"><parameter name="file" type="filename" value="${f("albedo")}"/></image>
      <image name="tex_normal" type="vector3"><parameter name="file" type="filename" value="${f("normal")}"/></image>
      <image name="tex_roughness" type="float"><parameter name="file" type="filename" value="${f("roughness")}"/></image>
      <image name="tex_metalness" type="float"><parameter name="file" type="filename" value="${f("metallic")}"/></image>
      <image name="tex_ao" type="float"><parameter name="file" type="filename" value="${f("ao")}"/></image>
      ${maps.cavityMap ? `<image name="tex_cavity" type="float"><parameter name="file" type="filename" value="${f("cavityMap")}"/></image>` : ""}
      ${maps.edgeWearMap ? `<image name="tex_edge" type="float"><parameter name="file" type="filename" value="${f("edgeWearMap")}"/></image>` : ""}
      ${maps.ccMask ? `<image name="tex_cc" type="float"><parameter name="file" type="filename" value="${f("ccMask")}"/></image>` : ""}
      ${maps.sheenMask ? `<image name="tex_sheen" type="color3"><parameter name="file" type="filename" value="${f("sheenMask")}"/></image>` : ""}
      ${maps.transMask ? `<image name="tex_trans" type="float"><parameter name="file" type="filename" value="${f("transMask")}"/></image>` : ""}
      <multiply name="color_with_ao" type="color3"><input name="in1" type="color3" nodename="tex_color"/><input name="in2" type="float" nodename="tex_ao"/></multiply>
    `;
    let roughNode = "tex_roughness";
    if (maps.cavityMap) {
      nodesXML += `<multiply name="cavity_mult" type="float"><input name="in1" type="float" nodename="tex_cavity"/><input name="in2" type="float" value="0.5"/></multiply><add name="rough_with_cavity" type="float"><input name="in1" type="float" nodename="tex_roughness"/><input name="in2" type="float" nodename="cavity_mult"/></add><clamp name="final_roughness" type="float"><input name="in" type="float" nodename="rough_with_cavity"/></clamp>`;
      roughNode = "final_roughness";
    }
    let coatNode = maps.edgeWearMap ? "tex_edge" : maps.ccMask ? "tex_cc" : "";
    let surfaceXML = `
      <standard_surface name="SRV_${p}" type="surfaceshader">
        <input name="base_color" type="color3" nodename="color_with_ao"/>
        <input name="normal" type="vector3" nodename="tex_normal"/>
        <input name="specular_roughness" type="float" nodename="${roughNode}"/>
        <input name="metalness" type="float" nodename="tex_metalness"/>
        ${coatNode ? `<input name="coat" type="float" nodename="${coatNode}"/>` : ""}
        ${maps.sheenMask ? `<input name="sheen_color" type="color3" nodename="tex_sheen"/>` : ""}
        ${maps.transMask ? `<input name="transmission" type="float" nodename="tex_trans"/>` : ""}
      </standard_surface>
    `;
    const mtlx = `<?xml version="1.0"?><materialx version="1.38">${nodesXML}${surfaceXML}<surfacematerial name="MAT_${p}" type="material"><input name="surfaceshader" type="surfaceshader" nodename="SRV_${p}"/></surfacematerial></materialx>`;
    const blob = new Blob([mtlx], { type: "application/xml" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${p}.mtlx`;
    a.click();
  };

  const handleSavePreset = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(
      new Blob(
        [JSON.stringify({ ...settings, quality: settings.quality }, null, 2)],
        { type: "application/json" },
      ),
    );
    a.download = `${pName()}_preset.json`;
    a.click();
  };
  const handleLoadPreset = () => {
    const i = document.createElement("input");
    i.type = "file";
    i.accept = ".json";
    i.onchange = (e) => {
      const r = new FileReader();
      r.onload = (ev) => {
        try {
          const p = JSON.parse(ev.target.result);
          setSettings((s) => ({ ...s, ...p }));
        } catch {
          alert("Invalid JSON");
        }
      };
      r.readAsText(e.target.files[0]);
    };
    i.click();
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
        *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
        html,body,#root{height:100vh;width:100vw;max-width:none;margin:0;padding:0;overflow:hidden;background:${C.bg}}
        body{font-family:'Inter',sans-serif;font-size:12px;color:${C.text};-webkit-font-smoothing:antialiased}
        input[type=range]{-webkit-appearance:none;width:100%;height:3px;background:${C.panel};outline:none;border-radius:2px}
        input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:${C.accent};cursor:pointer;}
        @keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
        button{font-family:'Inter',sans-serif;cursor:pointer;transition:all .15s}
        button:hover:not(:disabled){filter:brightness(1.15)}
        select{appearance:none; background:${C.panel}; border:1px solid ${C.border}; color:${C.text}; padding:4px; border-radius:4px; font-size:11px; outline:none;}
      `}</style>

      <div
        style={{
          display: "flex",
          width: "100vw",
          height: "100vh",
          overflow: "hidden",
        }}
      >
        <aside
          style={{
            width: 340,
            borderRight: `1px solid ${C.border}`,
            background: C.sidebar,
            display: "flex",
            flexDirection: "column",
            overflowY: "auto",
            flexShrink: 0,
          }}
        >
          <div style={{ padding: 16, borderBottom: `1px solid ${C.border}` }}>
            <div
              style={{
                fontSize: 9,
                letterSpacing: 3,
                color: C.textDim,
                fontFamily: "'JetBrains Mono'",
              }}
            >
              UNARRIVED
            </div>
            <div style={{ fontSize: 20, fontWeight: 600, color: C.accent }}>
              MAPPER <span style={{ fontSize: 10, color: C.red }}>v9.8</span>
            </div>
          </div>

          <div style={{ padding: 12 }}>
            <div
              onClick={() => {
                const i = document.createElement("input");
                i.type = "file";
                i.accept = "image/*";
                i.onchange = (e) => loadFile(e.target.files[0]);
                i.click();
              }}
              style={{
                border: `1px dashed ${C.border}`,
                borderRadius: 6,
                cursor: "pointer",
                background: C.panel,
                textAlign: "center",
                overflow: "hidden",
              }}
            >
              {srcURL ? (
                <img
                  src={srcURL}
                  style={{ width: "100%", height: 80, objectFit: "cover" }}
                />
              ) : (
                <div style={{ padding: 20, color: C.textDim }}>
                  ⬡ Drop Image
                </div>
              )}
            </div>
          </div>

          <div
            style={{
              padding: "0 12px 12px",
              borderBottom: `1px solid ${C.border}`,
              display: "flex",
              flexDirection: "column",
              gap: 6,
            }}
          >
            <Toggle
              label="Seamless Tiling (FBM)"
              value={settings.seamless}
              onChange={tg("seamless")}
              color={C.green}
            />
            <Toggle
              label="HQ Segmentation (b2)"
              value={settings.quality}
              onChange={tg("quality")}
              color={C.purple}
            />
            <Toggle
              label="Procedural Damage Mode"
              value={settings.metalMode}
              onChange={tg("metalMode")}
              color={C.accent}
            />
          </div>

          <div
            style={{
              padding: "6px 12px",
              borderBottom: `1px solid ${C.border}`,
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <div
              style={{
                width: 6,
                height: 6,
                borderRadius: 3,
                background: S[aiStep].d,
                animation: proc ? "blink 1s infinite" : "none",
              }}
            />
            <div style={{ fontSize: 11, color: C.textDim }}>
              {tileProg ? `Tiling ${tileProg.c}/${tileProg.t}` : S[aiStep].l}
            </div>
          </div>

          <div
            style={{
              padding: 12,
              display: "flex",
              flexDirection: "column",
              gap: 16,
            }}
          >
            <Section label="🎨 Base Color">
              <Slider
                label="Brightness"
                v={settings.brightness}
                min={0}
                max={2}
                step={0.05}
                onChange={ss("brightness")}
                onReset={sr("brightness")}
              />
              <Slider
                label="Contrast"
                v={settings.albContrast}
                min={0}
                max={2}
                step={0.05}
                onChange={ss("albContrast")}
                onReset={sr("albContrast")}
              />
              <Slider
                label="Saturation"
                v={settings.saturation}
                min={0}
                max={2}
                step={0.05}
                onChange={ss("saturation")}
                onReset={sr("saturation")}
              />
            </Section>

            <Section label="⛰️ Form & Depth">
              <Slider
                label="Height Mid-Level"
                v={settings.dispMid}
                min={0}
                max={1}
                step={0.01}
                onChange={ss("dispMid")}
                onReset={sr("dispMid")}
              />
              <Slider
                label="Depth Blur (Denoise)"
                v={settings.blurRadius}
                min={0}
                max={20}
                step={0.5}
                onChange={ss("blurRadius")}
                onReset={sr("blurRadius")}
              />
              <div
                style={{
                  marginTop: 6,
                  padding: "8px",
                  background: C.panel,
                  borderRadius: 4,
                }}
              >
                <div
                  style={{ fontSize: 10, color: C.textDim, marginBottom: 6 }}
                >
                  Ambient Occlusion
                </div>
                <Slider
                  label="AO Radius"
                  v={settings.aoRadius}
                  min={0.1}
                  max={3.0}
                  step={0.1}
                  onChange={ss("aoRadius")}
                  onReset={sr("aoRadius")}
                />
                <Slider
                  label="AO Intensity"
                  v={settings.aoStr}
                  min={0}
                  max={3.0}
                  step={0.1}
                  onChange={ss("aoStr")}
                  onReset={sr("aoStr")}
                />
              </div>
            </Section>

            <Section label="🔍 Surface Detail (Normal)">
              <Slider
                label="Macro Normal Str"
                v={settings.normalStr}
                min={0.5}
                max={10}
                step={0.1}
                onChange={ss("normalStr")}
                onReset={sr("normalStr")}
              />
              <Slider
                label="Bilateral Filter (σ)"
                v={settings.normalSigma}
                min={0.01}
                max={0.3}
                step={0.01}
                onChange={ss("normalSigma")}
                onReset={sr("normalSigma")}
              />
              <Slider
                label="Micro Detail Mix"
                v={settings.detailMix}
                min={0}
                max={1}
                step={0.01}
                onChange={ss("detailMix")}
                onReset={sr("detailMix")}
              />
              <div style={{ marginTop: 4 }}>
                <Toggle
                  label="Invert Normal Y (DirectX)"
                  value={settings.normalInvertY}
                  onChange={tg("normalInvertY")}
                  color={C.blue}
                />
              </div>

              <div
                style={{
                  marginTop: 8,
                  paddingTop: 8,
                  borderTop: `1px dashed ${C.border}`,
                }}
              >
                <div
                  style={{ fontSize: 10, color: C.textDim, marginBottom: 6 }}
                >
                  Micro-Surface Type (Hair/Leather/etc)
                </div>
                <select
                  value={settings.microType}
                  onChange={ss("microType")}
                  style={{ width: "100%", marginBottom: 6 }}
                >
                  <option value={0}>None (Smooth)</option>
                  <option value={1}>
                    Fine Grain / Sand / Fur (모래/미세입자)
                  </option>
                  <option value={2}>Leather / Skin Pores (가죽/모공)</option>
                  <option value={3}>Fabric / Canvas Weave (직물/캔버스)</option>
                  <option value={4}>Brushed Metal (한방향 갈아낸 철결)</option>
                  <option value={5}>Hammered Iron (망치로 두드린 철판)</option>
                  <option value={6}>
                    Cracks & Veins (날카로운 갈라짐/크랙)
                  </option>
                </select>
                {settings.microType > 0 && (
                  <>
                    <Slider
                      label="Micro Scale"
                      v={settings.microScale}
                      min={10}
                      max={1500}
                      step={10}
                      onChange={ss("microScale")}
                      onReset={sr("microScale")}
                    />
                    <Slider
                      label="Micro Depth"
                      v={settings.microDepth}
                      min={0}
                      max={1}
                      step={0.01}
                      onChange={ss("microDepth")}
                      onReset={sr("microDepth")}
                    />
                  </>
                )}
              </div>
            </Section>

            <Section label="💥 Wear & Tear">
              {settings.metalMode && (
                <>
                  <Slider
                    label="Rust Oxidation"
                    v={settings.rustLvl}
                    min={0}
                    max={1}
                    step={0.01}
                    onChange={ss("rustLvl")}
                    onReset={sr("rustLvl")}
                  />
                  <Slider
                    label="Surface Scratches"
                    v={settings.scratchLvl}
                    min={0}
                    max={1}
                    step={0.01}
                    onChange={ss("scratchLvl")}
                    onReset={sr("scratchLvl")}
                  />
                </>
              )}
              <Slider
                label="Curvature Scale"
                v={settings.curvScale}
                min={0.5}
                max={5}
                step={0.1}
                onChange={ss("curvScale")}
                onReset={sr("curvScale")}
              />
              <Slider
                label="Edge Wear Opacity"
                v={settings.edgeWear}
                min={0}
                max={1}
                step={0.01}
                onChange={ss("edgeWear")}
                onReset={sr("edgeWear")}
              />
              <Slider
                label="Edge Wear Contrast"
                v={settings.edgeContrast}
                min={0.5}
                max={5}
                step={0.1}
                onChange={ss("edgeContrast")}
                onReset={sr("edgeContrast")}
              />
              <Slider
                label="Surface Grunge"
                v={settings.grungeLvl}
                min={0}
                max={1}
                step={0.01}
                onChange={ss("grungeLvl")}
                onReset={sr("grungeLvl")}
              />
            </Section>

            <Section label="💡 PBR Properties">
              <div
                style={{
                  padding: "0 0 8px 0",
                  borderBottom: `1px dashed ${C.border}`,
                  marginBottom: 8,
                }}
              >
                <Slider
                  label="Roughness Contrast"
                  v={settings.roughCon}
                  min={0.5}
                  max={5}
                  step={0.1}
                  onChange={ss("roughCon")}
                  onReset={sr("roughCon")}
                />
                <Slider
                  label="Roughness Bias"
                  v={settings.roughBias}
                  min={-1}
                  max={1}
                  step={0.05}
                  onChange={ss("roughBias")}
                  onReset={sr("roughBias")}
                />

                <div
                  style={{
                    marginTop: 6,
                    padding: "8px",
                    background: C.panel,
                    borderRadius: 4,
                  }}
                >
                  <Slider
                    label="Roughness Min"
                    v={settings.roughMin}
                    min={0}
                    max={1}
                    step={0.01}
                    onChange={ss("roughMin")}
                    onReset={sr("roughMin")}
                  />
                  <Slider
                    label="Roughness Max"
                    v={settings.roughMax}
                    min={0}
                    max={1}
                    step={0.01}
                    onChange={ss("roughMax")}
                    onReset={sr("roughMax")}
                  />
                </div>

                <Slider
                  label="Material Sync Blend"
                  v={settings.matBlend}
                  min={0}
                  max={1}
                  step={0.01}
                  onChange={ss("matBlend")}
                  onReset={sr("matBlend")}
                />
                <Slider
                  label="High-Freq Roughness"
                  v={settings.roughDetailMix}
                  min={0}
                  max={1}
                  step={0.01}
                  onChange={ss("roughDetailMix")}
                  onReset={sr("roughDetailMix")}
                />
                <div style={{ marginTop: 4 }}>
                  <Toggle
                    label="Invert Roughness (Gloss)"
                    value={settings.roughInv}
                    onChange={tg("roughInv")}
                    color={C.purple}
                  />
                </div>
              </div>
              <Slider
                label="Metallic Threshold"
                v={settings.metalThr}
                min={0}
                max={1}
                step={0.01}
                onChange={ss("metalThr")}
                onReset={sr("metalThr")}
              />
              <Slider
                label="Metallic Softness"
                v={settings.metalSoft}
                min={0.01}
                max={0.3}
                step={0.01}
                onChange={ss("metalSoft")}
                onReset={sr("metalSoft")}
              />
            </Section>
          </div>

          <div
            style={{
              padding: 12,
              borderTop: `1px solid ${C.border}`,
              marginTop: "auto",
              display: "flex",
              gap: 4,
            }}
          >
            <button
              onClick={handleSavePreset}
              style={{
                flex: 1,
                padding: "6px 0",
                fontSize: 10,
                background: C.panel,
                border: `1px solid ${C.border}`,
                color: C.textDim,
              }}
            >
              ↓ Save Preset
            </button>
            <button
              onClick={handleLoadPreset}
              style={{
                flex: 1,
                padding: "6px 0",
                fontSize: 10,
                background: C.panel,
                border: `1px solid ${C.border}`,
                color: C.textDim,
              }}
            >
              ↑ Load Preset
            </button>
          </div>
        </aside>

        <main
          style={{
            flex: 1,
            minWidth: 0,
            background: C.bg,
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
            alignContent: "start",
            gap: 4,
            padding: 4,
            overflowY: "auto",
          }}
        >
          {MAP_DISPLAY.map((m) => (
            <div
              key={m.k}
              style={{
                aspectRatio: "1/1",
                background: "#09090f",
                position: "relative",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                borderRadius: 4,
                overflow: "hidden",
                border: `1px solid ${C.border}`,
              }}
            >
              {maps?.[m.k] ? (
                <img
                  src={maps[m.k]}
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "contain",
                  }}
                />
              ) : (
                <div style={{ fontSize: 11, color: C.textDim }}>{m.l}</div>
              )}
              <div
                style={{
                  position: "absolute",
                  top: 8,
                  left: 8,
                  fontSize: 10,
                  fontWeight: 600,
                  color: m.c,
                  background: "rgba(0,0,0,0.6)",
                  padding: "2px 6px",
                  borderRadius: 4,
                }}
              >
                {m.l}
              </div>
            </div>
          ))}
        </main>

        <aside
          style={{
            width: 320,
            borderLeft: `1px solid ${C.border}`,
            background: C.sidebar,
            display: "flex",
            flexDirection: "column",
            flexShrink: 0,
          }}
        >
          <div
            style={{
              width: "100%",
              aspectRatio: "1/1",
              position: "relative",
              background: "#000",
              borderBottom: `1px solid ${C.border}`,
            }}
          >
            <canvas
              ref={cvs}
              style={{ width: "100%", height: "100%", cursor: "grab" }}
            />
          </div>
          <div
            style={{
              padding: 16,
              borderBottom: `1px solid ${C.border}`,
              display: "flex",
              flexDirection: "column",
              gap: 10,
            }}
          >
            <div style={{ display: "flex", gap: 4 }}>
              {[
                { k: "sphere", l: "Sphere" },
                { k: "plane", l: "Plane" },
                { k: "torus", l: "Torus" },
              ].map((s) => (
                <button
                  key={s.k}
                  onClick={() => setSettings((st) => ({ ...st, shape: s.k }))}
                  style={{
                    flex: 1,
                    padding: "6px 0",
                    fontSize: 11,
                    background:
                      settings.shape === s.k ? C.accent + "22" : "transparent",
                    border: `1px solid ${settings.shape === s.k ? C.accent : C.border}`,
                    color: settings.shape === s.k ? C.accent : C.textDim,
                  }}
                >
                  {s.l}
                </button>
              ))}
            </div>

            <div
              style={{
                padding: "8px",
                background: C.panel,
                borderRadius: 4,
                marginTop: 4,
              }}
            >
              <div style={{ fontSize: 10, color: C.textDim, marginBottom: 6 }}>
                Viewport Lighting
              </div>
              <Slider
                label="Environment Brightness"
                v={settings.envExposure}
                min={0.1}
                max={3.0}
                step={0.1}
                onChange={ss("envExposure")}
                onReset={sr("envExposure")}
              />
              <div style={{ marginTop: 8 }}>
                <Slider
                  label="Lighting Rotation"
                  v={settings.envRotation}
                  min={0}
                  max={Math.PI * 2}
                  step={0.1}
                  onChange={ss("envRotation")}
                  onReset={sr("envRotation")}
                />
              </div>
            </div>

            <Slider
              label="Displacement Scale"
              v={settings.dispScale}
              min={0}
              max={0.5}
              step={0.01}
              onChange={ss("dispScale")}
              onReset={sr("dispScale")}
            />

            <div style={{ display: "flex", gap: 4, marginTop: 4 }}>
              {[1, 2, 3, 4].map((n) => (
                <button
                  key={n}
                  onClick={() => setSettings((s) => ({ ...s, tileRep: n }))}
                  style={{
                    flex: 1,
                    padding: "4px 0",
                    fontSize: 10,
                    background:
                      settings.tileRep === n ? C.blue + "22" : "transparent",
                    border: `1px solid ${settings.tileRep === n ? C.blue : C.border}`,
                    color: settings.tileRep === n ? C.blue : C.textDim,
                  }}
                >
                  {n}× Tile
                </button>
              ))}
            </div>
          </div>

          <div
            style={{
              padding: 16,
              marginTop: "auto",
              display: "flex",
              flexDirection: "column",
              gap: 8,
            }}
          >
            <div
              style={{
                fontSize: 10,
                fontWeight: 600,
                color: C.textDim,
                textTransform: "uppercase",
                marginBottom: 4,
              }}
            >
              Export Resolution
            </div>
            <div style={{ display: "flex", gap: 4, marginBottom: 8 }}>
              {[512, 1024, 2048, 4096].map((r) => (
                <button
                  key={r}
                  onClick={() => setSettings((s) => ({ ...s, res: r }))}
                  style={{
                    flex: 1,
                    padding: "4px 0",
                    fontSize: 10,
                    background:
                      settings.res === r ? C.accent + "22" : "transparent",
                    border: `1px solid ${settings.res === r ? C.accent : C.border}`,
                    color: settings.res === r ? C.accent : C.textDim,
                  }}
                >
                  {r >= 1000 ? `${r / 1024}K` : r}
                </button>
              ))}
            </div>
            <button
              onClick={handleGLB}
              disabled={!maps || exporting}
              style={{
                width: "100%",
                padding: "12px 0",
                borderRadius: 6,
                fontSize: 12,
                fontWeight: 600,
                background: maps ? C.accent : C.panel,
                border: `1px solid ${maps ? C.accent : C.border}`,
                color: maps ? C.bg : C.textDim,
              }}
            >
              {exporting ? `Baking Displacement...` : `🧊 Export GLB`}
            </button>
            <div style={{ display: "flex", gap: 4 }}>
              <button
                onClick={handleZip}
                disabled={!maps}
                style={{
                  flex: 1,
                  padding: "8px 0",
                  borderRadius: 6,
                  fontSize: 11,
                  background: C.panel,
                  border: `1px solid ${C.border}`,
                  color: C.text,
                }}
              >
                📦 ZIP Textures
              </button>
              <button
                onClick={handleMTLX}
                disabled={!maps}
                style={{
                  flex: 1,
                  padding: "8px 0",
                  borderRadius: 6,
                  fontSize: 11,
                  background: C.panel,
                  border: `1px solid ${C.border}`,
                  color: C.text,
                }}
              >
                ⚙️ MaterialX
              </button>
            </div>
          </div>
        </aside>
      </div>
    </>
  );
}

const Section = ({ label, children }) => (
  <div>
    <div
      style={{
        fontSize: 10,
        fontWeight: 600,
        color: C.textDim,
        textTransform: "uppercase",
        marginBottom: 8,
        letterSpacing: 0.5,
      }}
    >
      {label}
    </div>
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {children}
    </div>
  </div>
);
// 🔥 [UX 개선] 더블클릭 초기화 연결 (userSelect:"none" 추가로 드래그 선택 방지)
const Slider = ({ label, v, min, max, step, onChange, onReset }) => (
  <div>
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        fontSize: 11,
        color: C.text,
        marginBottom: 4,
      }}
    >
      <span>{label}</span>
      <span
        onDoubleClick={onReset}
        title="Double click to reset"
        style={{
          color: C.accent,
          fontFamily: "'JetBrains Mono', monospace",
          cursor: onReset ? "pointer" : "default",
          userSelect: "none",
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
  </div>
);
const Toggle = ({ label, value, onChange, color }) => (
  <div
    style={{
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      fontSize: 11,
      color: C.text,
    }}
  >
    {label}
    <div
      onClick={onChange}
      style={{
        width: 28,
        height: 16,
        borderRadius: 8,
        cursor: "pointer",
        position: "relative",
        background: value ? color : C.panel,
        border: `1px solid ${value ? color : C.border}`,
        transition: "all .2s",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 1,
          left: value ? 13 : 1,
          width: 12,
          height: 12,
          borderRadius: 6,
          background: value ? "#fff" : C.textDim,
          transition: "all .2s",
        }}
      />
    </div>
  </div>
);
