import { useState, useRef, useEffect, useCallback } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFExporter } from "three/examples/jsm/exporters/GLTFExporter.js";
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
      { device: "wasm" },
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
   GPU QUERY & UTILS
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
const randomSeed = () => Math.random() * 1000.0;

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
  tex.colorSpace = THREE.SRGBColorSpace;
  const pm = new THREE.PMREMGenerator(renderer);
  pm.compileEquirectangularShader();
  const env = pm.fromEquirectangular(tex).texture;
  pm.dispose();
  tex.dispose();
  return env;
}

/* ═══════════════════════════════════════════════════════════════
   MATERIAL CLASS MASKING
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
   GLSL3 SHADERS & ALGORITHMS (FULL CONTROL RESTORED)
═══════════════════════════════════════════════════════════════ */
const GLSL_COMMON = `
  uniform float seed;
  vec2 sUv(vec2 uv) { return uv + vec2(fract(sin(seed)*100.0), fract(cos(seed)*100.0)) * 100.0; }
  float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233))) * 43758.5453); }
  vec2 hash2(vec2 p){ p=vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))); return fract(sin(p)*43758.5453); }
  float noise(vec2 p){ 
    vec2 i=floor(p); vec2 f=fract(p); vec2 u=f*f*(3.0-2.0*f); 
    return mix(mix(hash(i),hash(i+vec2(1,0)),u.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),u.x),u.y); 
  } 
  float fbm(vec2 p){ 
    float v=0.,a=0.5; vec2 shift=vec2(100.); mat2 rot=mat2(cos(0.5),sin(0.5),-sin(0.5),cos(0.5)); 
    for(int i=0;i<5;++i){ v+=a*noise(p); p=rot*p*2.+shift; a*=0.5; } return v; 
  }
  float voronoi(vec2 x) {
    vec2 n = floor(x); vec2 f = fract(x); float m = 8.0;
    for(int j=-1; j<=1; j++) for(int i=-1; i<=1; i++) {
      vec2 g = vec2(float(i), float(j));
      vec2 o = hash2(n + g);
      vec2 r = g - f + o;
      float d = dot(r,r);
      m = min(m, d);
    }
    return sqrt(m);
  }
  float scratch(vec2 uv, float angle, float density) {
    float s = sin(angle), c = cos(angle);
    vec2 rotUv = vec2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);
    rotUv.x *= density; rotUv.y *= 2.0;
    float n = noise(rotUv * 20.0);
    return smoothstep(0.85, 0.95, n);
  }
`;

const VS = `#version 300 es\nin vec2 a;out vec2 v;\nvoid main(){v=a*.5+.5;gl_Position=vec4(a,0.,1.);}`;
const FS_PASS = `#version 300 es\nprecision highp float;\nuniform sampler2D T;in vec2 v;out vec4 o;\nvoid main(){o=texture(T,v);}`;
const FS_CHANNEL = `#version 300 es\nprecision highp float; uniform sampler2D T; uniform float ch; in vec2 v; out vec4 o; void main(){ float c=0.0; if(ch<0.5) c=texture(T,v).r; else if(ch<1.5) c=texture(T,v).g; else if(ch<2.5) c=texture(T,v).b; else c=texture(T,v).a; o=vec4(c,c,c,1.0); }`;

const FS_COLOR = `#version 300 es
precision highp float; uniform sampler2D T; uniform float brightness, contrast, saturation; in vec2 v; out vec4 o;
void main(){
  vec3 c=texture(T,v).rgb; c=(c-0.5)*contrast+0.5;
  float lum=dot(c,vec3(0.2126,0.7152,0.0722)); c=mix(vec3(lum),c,saturation); c*=brightness;
  o=vec4(clamp(c,0.0,1.0),1.0);
}`;

const FS_SEAMLESS = `#version 300 es\nprecision highp float; uniform sampler2D T; in vec2 v; out vec4 o; ${GLSL_COMMON} void main(){ vec2 uv=mod(v,1.0); vec2 uv2=mod(v+0.5,1.0); float n=fbm(sUv(v)*15.0)*0.15-0.075; float wx=smoothstep(0.2,0.8,abs(v.x-0.5)*2.0+n); float wy=smoothstep(0.2,0.8,abs(v.y-0.5)*2.0+n); float mask=max(wx,wy); o=mix(texture(T,uv),texture(T,uv2),mask); }`;

const FS_BH = `#version 300 es\nprecision highp float;\nuniform sampler2D T;uniform vec2 px;uniform float radius;in vec2 v;out vec4 o;const float W[5]=float[5](0.227027,0.194595,0.121622,0.054054,0.016216);void main(){vec4 c=texture(T,v)*W[0];for(int i=1;i<5;i++)c+=texture(T,v+vec2(float(i)*px.x*radius,0.))*W[i]+texture(T,v-vec2(float(i)*px.x*radius,0.))*W[i];o=c;}`;
const FS_BV = `#version 300 es\nprecision highp float;\nuniform sampler2D T;uniform vec2 px;uniform float radius;in vec2 v;out vec4 o;const float W[5]=float[5](0.227027,0.194595,0.121622,0.054054,0.016216);void main(){vec4 c=texture(T,v)*W[0];for(int i=1;i<5;i++)c+=texture(T,v+vec2(0.,float(i)*px.y*radius))*W[i]+texture(T,v-vec2(0.,float(i)*px.y*radius))*W[i];o=c;}`;
const FS_HP = `#version 300 es\nprecision highp float;\nuniform sampler2D ORIG,BLUR;in vec2 v;out vec4 o;void main(){o=clamp(texture(ORIG,v)-texture(BLUR,v)+vec4(.5),0.,1.);}`;

// 🔥 RAW 절차적 패턴 마스크 생성 (깊이 연산 분리)
const FS_PROC_HEIGHT = `#version 300 es
precision highp float; 
uniform float rustLvl, scratchLvl, rustScale, scratchScale; 
uniform float microType, microScale;
in vec2 v; out vec4 o; ${GLSL_COMMON}
void main(){
  vec2 suv = sUv(v); 
  
  // 1. Rust Coverage
  float rust = 1.0 - voronoi(suv * rustScale);
  float rustSpread = fbm(suv * (rustScale * 0.3));
  rust = smoothstep(0.4 - rustLvl*0.4, 0.9, rust * rustSpread * 2.0) * rustLvl;
  
  // 2. Scratch Coverage
  float s1 = scratch(suv, 0.5, scratchScale);
  float s2 = scratch(suv, -0.2, scratchScale * 0.8);
  float scr = max(s1, s2) * scratchLvl;
  
  // 3. Micro-Surface Pattern (Raw)
  float micro = 0.0; vec2 muv = suv * microScale;
  if (microType == 1.0) { micro = hash(muv); } 
  else if (microType == 2.0) { micro = pow(1.0 - voronoi(muv), 3.0); } 
  else if (microType == 3.0) { micro = (sin(muv.x) * sin(muv.y)) * 0.5 + 0.5; } 
  else if (microType == 4.0) { micro = noise(vec2(muv.x * 0.05, muv.y)); }

  o = vec4(rust, scr, micro, 1.0);
}`;

// 🔥 노멀 맵 - 슬라이더(Depth 파라미터) 연동 완벽 반영
const FS_NM = `#version 300 es\nprecision highp float; uniform sampler2D DEPTH,DETAIL,PROC; uniform vec2 px; 
uniform float str,detailMix,sigma,microDetail,invertY,metalMode; 
uniform float rustDepth, scratchDepth, microDepth;
in vec2 v;out vec4 o; ${GLSL_COMMON} 
float bilW(float dc,float ds){float d=dc-ds;return exp(-d*d/(sigma*sigma+.0001));} 
float hD(vec2 u){return texture(DEPTH,u).r;} 
float hA(vec2 u){ 
  float a = texture(DETAIL,u).r - 0.5;
  vec4 p = texture(PROC, u); 
  if(metalMode > 0.5) { 
    a -= p.r * rustDepth;      // 녹의 물리적 파임 강도
    a -= p.g * scratchDepth;   // 스크래치의 물리적 파임 강도
  } 
  a += p.b * microDepth;       // 마이크로 서페이스의 요철 강도
  return a;
} 
void main(){ 
  float dc=hD(v); 
  vec2 ofs[8];ofs[0]=px*vec2(-1,1);ofs[1]=px*vec2(0,1);ofs[2]=px*vec2(1,1); ofs[3]=px*vec2(-1,0);ofs[4]=px*vec2(1,0);ofs[5]=px*vec2(-1,-1);ofs[6]=px*vec2(0,-1);ofs[7]=px*vec2(1,-1); 
  float kx[8];kx[0]=-1.;kx[1]=0.;kx[2]=1.;kx[3]=-2.;kx[4]=2.;kx[5]=-1.;kx[6]=0.;kx[7]=1.; 
  float ky[8];ky[0]=1.;ky[1]=2.;ky[2]=1.;ky[3]=0.;ky[4]=0.;ky[5]=-1.;ky[6]=-2.;ky[7]=-1.; 
  float dx=0.,dy=0.,adx=0.,ady=0.; 
  for(int i=0;i<8;i++){float ds=hD(v+ofs[i]);float bw=bilW(dc,ds);float as=hA(v+ofs[i]);dx+=kx[i]*ds*bw;dy+=ky[i]*ds*bw;adx+=kx[i]*as;ady+=ky[i]*as;} 
  float nNoise=(fbm(sUv(v)*40.0)-0.5)*microDetail*0.15; // 기본 FBM 노이즈
  vec3 nD=normalize(vec3(dx*str+nNoise,dy*str+nNoise,1.)); 
  vec3 nA=normalize(vec3(adx*str*1.5,ady*str*1.5,1.)); 
  vec3 n=normalize(vec3(nD.xy+nA.xy*detailMix,nD.z)); 
  if(invertY>0.5){n.y=-n.y;} o=vec4(n*.5+.5,1.);
}`;

const FS_CURV = `#version 300 es\nprecision highp float;\nuniform sampler2D NM;uniform vec2 px;uniform float scale;in vec2 v;out vec4 o;float cs(float s){vec3 n=normalize(texture(NM,v).rgb*2.-1.);float d=0.;vec2 ds[4];ds[0]=vec2(px.x*s,0.);ds[1]=vec2(0.,px.y*s);ds[2]=vec2(-px.x*s,0.);ds[3]=vec2(0.,-px.y*s);for(int i=0;i<4;i++)d+=dot(n,normalize(texture(NM,v+ds[i]).rgb*2.-1.));return clamp(-(d/4.-1.)*scale*3.+.5,0.,1.);}void main(){float c=cs(1.)*.5+cs(4.)*.3+cs(12.)*.2;o=vec4(c,c,c,1.);}`;

const FS_MASKS = `#version 300 es
precision highp float; uniform sampler2D CURV, PROC; uniform float edgeWear, edgeContrast, grungeLvl; in vec2 v; out vec4 o; ${GLSL_COMMON}
void main() {
  vec4 p = texture(PROC, v); float rust = p.r; float scr = p.g;
  float curv = texture(CURV, v).r;
  float wear = clamp((curv - 0.5) * 4.0, 0.0, 1.0);
  float n = fbm(sUv(v) * vec2(8.0, 12.0));
  float grunge = clamp((n - 0.5 + grungeLvl) * 2.5, 0.0, 1.0);
  wear = clamp((wear - (1.0 - edgeWear)) * edgeContrast, 0.0, 1.0);
  wear = wear * grunge;
  o = vec4(rust, scr, wear, grunge);
}`;

const FS_ROUGH = `#version 300 es\nprecision highp float; uniform sampler2D DEPTH,DETAIL,SEG,MASKS,PROC; uniform vec2 px; 
uniform float con,bias,inv,detailMix,matBlend,metalMode, microDepth; in vec2 v; out vec4 o; 
void main(){ 
  float sd=0.,sd2=0.; for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++){float s=texture(DEPTH,v+px*vec2(float(dx),float(dy))).r;sd+=s;sd2+=s*s;} sd/=25.;sd2/=25.;float lv=max(sd2-sd*sd,0.)*80.; 
  float fe=length(texture(DETAIL,v).rgb-vec3(.5))*2.; float lb=texture(DEPTH,v).r;float l=mix(lb,mix(lv,fe,.5),detailMix); 
  
  vec4 m = texture(MASKS, v); float rust = m.r, scr = m.g, wear = m.b, grunge = m.a;
  vec4 p = texture(PROC, v); float micro = p.b;

  l = clamp(l + grunge * 0.4, 0.0, 1.0); // 베이스 그런지
  
  if(metalMode > 0.5) { 
    l = mix(l, 0.98, rust); // 녹슨 부분의 난반사(완전 거칠게)
    l = mix(l, 0.15, max(scr, wear)); // 긁힌 부분의 정반사(매끄럽게)
  }
  
  l = clamp(l + micro * microDepth * 0.8, 0.0, 1.0); // 마이크로 패턴에 의한 난반사 추가
  l = mix(l, 1.-l, inv); 
  o = vec4(vec3(clamp((l-.5)*con+.5+bias,0.,1.)),1.);
}`;

const FS_METAL = `#version 300 es\nprecision highp float; uniform sampler2D T,SEG,MASKS; uniform float thr,soft,metalMode; in vec2 v; out vec4 o; 
void main(){ 
  float l=texture(T,v).r;
  float mt=smoothstep(thr-soft,thr+soft,l); 
  vec4 m = texture(MASKS, v); float rust = m.r, scr = m.g, wear = m.b;
  if(metalMode > 0.5) { 
    mt = 1.0; // 철 기반
    mt = mix(mt, 0.0, rust); // 녹슨 곳은 비금속
    mt = mix(mt, 1.0, max(scr, wear)); // 속살은 금속
  } else { 
    mt = clamp(mt + wear * 0.8, 0.0, 1.0); 
  }
  o=vec4(mt,mt,mt,1.);
}`;

const FS_DISP = `#version 300 es\nprecision highp float;\nuniform sampler2D T;uniform float mid,con,inv;in vec2 v;out vec4 o;void main(){float d=texture(T,v).r;d=clamp((d-mid)*con+.5,0.,1.);d=mix(d,1.-d,inv);o=vec4(d,d,d,1.);}`;
const FS_AO = `#version 300 es\nprecision highp float;\nuniform sampler2D T;uniform vec2 px;uniform float radius,strength;in vec2 v;out vec4 o;const float G=2.3999632;const int S=32;void main(){float dc=texture(T,v).r,occ=0.,dMn=1.,dMx=0.;for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++){float s=texture(T,v+px*vec2(float(dx),float(dy))*4.).r;dMn=min(dMn,s);dMx=max(dMx,s);}float dr=max(dMx-dMn,.01),ar=radius/dr;for(int i=0;i<S;i++){float r=sqrt(float(i+1)/float(S)),th=float(i)*G;vec2 off=vec2(cos(th),sin(th))*px*min(ar,radius)*80.*r;float diff=clamp((dc-texture(T,v+off).r)*8.,0.,1.);float cosW=(1.-r)*.7+.3;occ+=diff*cosW;}float ao=clamp(1.-(occ/float(S)*2.)*strength,0.,1.);o=vec4(ao,ao,ao,1.);}`;
const FS_ORM = `#version 300 es\nprecision highp float;\nuniform sampler2D AO,RO,ME;in vec2 v;out vec4 o;void main(){o=vec4(texture(AO,v).r,texture(RO,v).r,texture(ME,v).r,1.);}`;

// 🔥 다이나믹 흑철 산화(Oxidation) 알고리즘
const FS_ALBEDO_FINAL = `#version 300 es
precision highp float; uniform sampler2D T, MASKS, PROC; 
uniform float metalMode, seed, microDepth; in vec2 v; out vec4 o; ${GLSL_COMMON}
void main(){
  vec4 c = texture(T, v);
  vec4 p = texture(PROC, v); float micro = p.b;
  
  // 마이크로 오클루전 (미세 틈새 그림자)
  c.rgb *= (1.0 - micro * microDepth * 0.6); 

  if(metalMode > 0.5) {
    vec4 m = texture(MASKS, v); float rust = m.r, scr = m.g, wear = m.b;
    float lum = dot(c.rgb, vec3(0.2126,0.7152,0.0722));
    
    // 깊이 파인 곳은 짙은 고동색, 가장자리는 밝은 오렌지색 녹
    vec3 rustColorDark = vec3(0.18, 0.07, 0.03);
    vec3 rustColorLight = vec3(0.55, 0.28, 0.10);
    float rustVariation = fbm(sUv(v) * 25.0);
    vec3 finalRust = mix(rustColorDark, rustColorLight, rustVariation);
    
    // 바탕색을 어둡게 산화된 흑철(Dark Iron) 색상으로 톤다운시켜 묵직함 부여
    c.rgb = mix(c.rgb, vec3(0.15, 0.16, 0.18) * (lum + 0.2), 0.85);
    
    c.rgb = mix(c.rgb, finalRust, rust); // 녹 색상 합성
    c.rgb = mix(c.rgb, vec3(0.75, 0.8, 0.85), max(scr, wear)); // 스크래치는 빛나는 은색 메탈
  }
  o = vec4(clamp(c.rgb, 0.0, 1.0), c.a);
}`;

/* ═══════════════════════════════════════════════════════════════
   WebGL2 PIPELINE EXECUTION
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
    for (const [type, src] of [
      [gl.VERTEX_SHADER, VS],
      [gl.FRAGMENT_SHADER, fs],
    ]) {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS))
        console.error(
          "[Shader compile]",
          gl.getShaderInfoLog(sh),
          "\n",
          src.slice(0, 200),
        );
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
      if (u === null) continue;
      Array.isArray(v) ? gl.uniform2fv(u, v) : gl.uniform1f(u, v);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo ?? null);
    gl.viewport(0, 0, W, H);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  };

  const T = {
    origA: mkt(null),
    a: mkt(null),
    d: mkt(null),
    s: mkt(null),
    sa: mkt(null),
    sd: mkt(null),
    bh: mkt(null),
    bv: mkt(null),
    hi: mkt(null),
    proc: mkt(null),
    nm: mkt(null),
    cv: mkt(null),
    masks: mkt(null),
    ao: mkt(null),
    ro: mkt(null),
    me: mkt(null),
    di: mkt(null),
    finalA: mkt(null),
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
  upt(T.origA, albC);
  upt(T.d, depC);
  upt(T.s, segC);

  const globalUns = { seed: s.seed };

  exec(P.color, F.a, [["T", T.origA]], {
    brightness: s.brightness,
    contrast: s.albContrast,
    saturation: s.saturation,
    ...globalUns,
  });

  if (s.seamless) {
    exec(P.hqSeam, F.sa, [["T", T.a]], { ...globalUns });
    exec(P.hqSeam, F.sd, [["T", T.d]], { ...globalUns });
  }
  const srcA = s.seamless ? T.sa : T.a;
  const srcD = s.seamless ? T.sd : T.d;

  exec(P.bh, F.bh, [["T", srcA]], { px, radius: s.blurRadius, ...globalUns });
  exec(P.bv, F.bv, [["T", T.bh]], { px, radius: s.blurRadius, ...globalUns });
  exec(
    P.hp,
    F.hi,
    [
      ["ORIG", srcA],
      ["BLUR", T.bv],
    ],
    { ...globalUns },
  );

  // 1. 순수 범위(Coverage) 마스크 렌더링
  exec(P.procHeight, F.proc, [], {
    rustLvl: s.metalMode ? s.rustLvl : 0.0,
    scratchLvl: s.metalMode ? s.scratchLvl : 0.0,
    rustScale: s.rustScale,
    scratchScale: s.scratchScale,
    microType: s.microType,
    microScale: s.microScale,
    ...globalUns,
  });

  // 2. 물리적 깊이(Depth) 반영하여 노멀맵 베이킹
  exec(
    P.nm,
    F.nm,
    [
      ["DEPTH", srcD],
      ["DETAIL", T.hi],
      ["PROC", T.proc],
    ],
    {
      px,
      str: s.normalStr,
      detailMix: s.detailMix,
      sigma: s.normalSigma,
      microDetail: s.microDetail,
      invertY: s.normalInvertY ? 1 : 0,
      metalMode: s.metalMode ? 1 : 0,
      rustDepth: s.rustDepth,
      scratchDepth: s.scratchDepth,
      microDepth: s.microDepth,
      ...globalUns,
    },
  );

  exec(P.curv, F.cv, [["NM", T.nm]], { px, scale: s.curvScale, ...globalUns });
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
      ...globalUns,
    },
  );
  exec(P.ao, F.ao, [["T", srcD]], {
    px,
    radius: s.aoRadius,
    strength: s.aoStr,
    ...globalUns,
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
      microDepth: s.microDepth,
      ...globalUns,
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
    {
      thr: s.metalThr,
      soft: s.metalSoft,
      metalMode: s.metalMode ? 1 : 0,
      ...globalUns,
    },
  );
  exec(P.disp, F.di, [["T", srcD]], {
    mid: s.dispMid,
    con: s.dispCon,
    inv: s.dispInv ? 1 : 0,
    ...globalUns,
  });
  exec(
    P.albedoFinal,
    F.finalA,
    [
      ["T", srcA],
      ["MASKS", T.masks],
      ["PROC", T.proc],
    ],
    { metalMode: s.metalMode ? 1 : 0, microDepth: s.microDepth, ...globalUns },
  );

  const snap = (prog, binds, uns) => {
    exec(prog, null, binds, uns);
    return c.toDataURL("image/png");
  };
  const snapCh = (t, ch) => {
    exec(P.channel, null, [["T", t]], { ch });
    return c.toDataURL("image/png");
  };

  return {
    albedo: snap(P.pass, [["T", T.finalA]], {}),
    normal: snap(P.pass, [["T", T.nm]], {}),
    displacement: snap(P.pass, [["T", T.di]], {}),
    roughness: snap(P.pass, [["T", T.ro]], {}),
    metallic: snap(P.pass, [["T", T.me]], {}),
    ao: snap(P.pass, [["T", T.ao]], {}),
    curvature: snap(P.pass, [["T", T.cv]], {}),
    mask_rust: s.metalMode ? snapCh(T.masks, 0.0) : null,
    mask_scratch: s.metalMode ? snapCh(T.masks, 1.0) : null,
    mask_edge: snapCh(T.masks, 2.0),
    mask_grunge: snapCh(T.masks, 3.0),
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
  const sc = res / Math.max(img.naturalWidth, img.naturalHeight);
  const W = Math.round(img.naturalWidth * sc),
    H = Math.round(img.naturalHeight * sc);
  const neutral =
    segCvs ||
    (() => {
      const c = document.createElement("canvas");
      c.width = c.height = 64;
      const ctx = c.getContext("2d");
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, 64, 64);
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
    "albedo",
    "normal",
    "displacement",
    "roughness",
    "metallic",
    "ao",
    "curvature",
    "mask_rust",
    "mask_scratch",
    "mask_edge",
    "mask_grunge",
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
        if (!r[k]) continue;
        const ti = new Image();
        ti.src = r[k];
        await new Promise((res) => {
          ti.onload = res;
          ti.onerror = res;
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
  for (const k of keys) if (outs[k]) out[k] = outs[k].toDataURL("image/png");
  if (!s.metalMode) {
    delete out.mask_rust;
    delete out.mask_scratch;
  }
  return out;
}

/* ═══════════════════════════════════════════════════════════════
   EXPORT & TOKENS
═══════════════════════════════════════════════════════════════ */
async function doZip(maps, res, prefix) {
  const z = new JSZip(),
    f = z.folder(`${prefix}_${res}px`);
  await Promise.all(
    Object.entries(maps).map(async ([k, u]) => {
      if (u) f.file(`${prefix}_${k}.png`, await (await fetch(u)).blob());
    }),
  );
  const b = await z.generateAsync({ type: "blob" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b);
  a.download = `${prefix}_PBR_${res}px.zip`;
  a.click();
  URL.revokeObjectURL(a.href);
}

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
    key: "albedo",
    label: "Base Color",
    color: C.green,
    hint: "Color corrected & Seamless",
  },
  {
    key: "normal",
    label: "Normal Map",
    color: C.blue,
    hint: "Bilateral depth + micro details",
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
    hint: "Freq + Material + Grunge",
  },
  {
    key: "metallic",
    label: "Metallic",
    color: C.teal,
    hint: "Threshold + Edge Wear",
  },
  {
    key: "ao",
    label: "Ambient Occ.",
    color: C.red,
    hint: "HBAO 32-sample spiral",
  },
];
const MAPS_SMART = [
  {
    key: "curvature",
    label: "Curvature",
    color: C.yellow,
    hint: "Multi-scale edge wear base",
  },
  {
    key: "mask_edge",
    label: "Edge Wear Mask",
    color: C.pink,
    hint: "Extracts exposed edges",
  },
  {
    key: "mask_grunge",
    label: "Grunge Mask",
    color: C.green,
    hint: "Procedural surface dirt",
  },
  {
    key: "mask_rust",
    label: "Rust Mask",
    color: "#c05020",
    hint: "Voronoi Pitting & Oxidation",
  },
  {
    key: "mask_scratch",
    label: "Scratch Mask",
    color: "#d0d0d0",
    hint: "Directional surface damage",
  },
];
const MAPS_EXTRA = [
  {
    key: "orm",
    label: "ORM Pack",
    color: C.blue,
    hint: "R:AO G:Rough B:Metal (UE/Unity)",
  },
];
const ALL_MAPS = [...MAPS_DEF, ...MAPS_SMART, ...MAPS_EXTRA];

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

// 🔥 모든 컨트롤 파라미터 복구 및 정교화
const DEF = {
  seed: 42.0,
  metalMode: false,
  seamless: false,
  brightness: 1.0,
  albContrast: 1.0,
  saturation: 1.0,

  microType: 0,
  microScale: 400,
  microDepth: 0.5,

  rustLvl: 0.6,
  rustScale: 15.0,
  rustDepth: 2.5, // Coverage vs Scale vs Depth 분리
  scratchLvl: 0.5,
  scratchScale: 50.0,
  scratchDepth: 1.5, // Coverage vs Scale vs Depth 분리

  edgeWear: 0.4,
  edgeContrast: 2.5,
  grungeLvl: 0.3,

  normalStr: 3,
  detailMix: 0.4,
  normalSigma: 0.05,
  microDetail: 0.4,
  normalInvertY: false, // microDetail(FBM) 복구

  dispMid: 0.5,
  dispCon: 2,
  dispInv: false,
  dispScale: 0.05,

  roughCon: 2,
  roughBias: 0,
  roughInv: false,
  roughDetailMix: 0.5,
  matBlend: 0.25, // 러프니스 디테일 복구

  metalThr: 0.75,
  metalSoft: 0.1, // 메탈릭 복구

  blurRadius: 6,
  aoRadius: 0.8,
  aoStr: 1.2,
  curvScale: 1,
  envRotation: 0,
};

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
  const [tileProg, setTileProg] = useState(null);
  const [quality, setQuality] = useState(false);

  const [shape, setShape] = useState("sphere");
  const [tileRep, setTileRep] = useState(1);
  const [viewMode, setViewMode] = useState("pbr");
  const [res, setRes] = useState(1024);
  const [hdri, setHdri] = useState("studio");
  const [gpu, setGpu] = useState({ maxTex: 4096, renderer: "Querying…" });

  const cvs = useRef(null),
    imgR = useRef(null),
    depR = useRef(null),
    segR = useRef(null);
  const setR = useRef(settings),
    resR = useRef(res),
    threeR = useRef({}),
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
        setImgInfo({
          w: img.naturalWidth,
          h: img.naturalHeight,
          name: file.name.split(".")[0],
        });
        setMaps(null);
        depR.current = null;
        segR.current = null;
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
          const isTile = resR.current > TILE;
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
    const tid = ++tok.current;
    const id = setTimeout(async () => {
      if (tid !== tok.current) return;
      setProc(true);
      setTileProg(null);
      const isTile = res > TILE;
      setAiStep(isTile ? "tiling" : "generating");
      try {
        const r = await generateMaps(
          imgR.current,
          depR.current,
          segR.current,
          settings,
          res,
          (c, t) => {
            if (tid === tok.current) setTileProg({ c, t });
          },
        );
        if (tid !== tok.current) return;
        setMaps(r);
        setAiStep("ready");
      } catch (err) {
        if (tid !== tok.current) return;
        console.error(err);
        setAiStep("error");
      } finally {
        if (tid === tok.current) {
          setTileProg(null);
          setProc(false);
        }
      }
    }, 300);
    return () => clearTimeout(id);
  }, [settings, res]);

  /* ═══════════════════════════════════════════════════════════════
     THREE.JS PRO PREVIEW SETUP
  ═══════════════════════════════════════════════════════════════ */
  useEffect(() => {
    const el = cvs.current;
    if (!el) return;
    const renderer = new THREE.WebGLRenderer({
      canvas: el,
      antialias: true,
      preserveDrawingBuffer: true,
    });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 50);
    camera.position.set(0, 0, 3.2);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    const env = buildEnv(renderer, "studio");
    scene.environment = env;
    scene.background = env;

    const light = new THREE.DirectionalLight(0xffffff, 1.5);
    light.position.set(4, 5, 3);
    light.castShadow = true;
    light.shadow.mapSize.width = 1024;
    light.shadow.mapSize.height = 1024;
    scene.add(light);
    scene.add(new THREE.AmbientLight(0xffffff, 0.2));

    const G = {
      sphere: new THREE.SphereGeometry(1, 256, 128),
      plane: new THREE.PlaneGeometry(2, 2, 256, 256),
      torus: new THREE.TorusGeometry(0.72, 0.34, 128, 256),
    };
    Object.values(G).forEach((g) => g.setAttribute("uv2", g.attributes.uv));

    const mat = new THREE.MeshStandardMaterial({
      roughness: 0.5,
      metalness: 0,
      color: 0xd0d0d0,
      envMapIntensity: 1,
    });
    const debugMat = new THREE.MeshBasicMaterial({ color: 0xffffff });

    const mesh = new THREE.Mesh(G.sphere, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    scene.add(mesh);

    const gm = new THREE.Mesh(
      new THREE.PlaneGeometry(20, 20),
      new THREE.MeshStandardMaterial({ color: 0x0a0a18, roughness: 1 }),
    );
    gm.rotation.x = -Math.PI / 2;
    gm.position.y = -1.4;
    gm.receiveShadow = true;
    scene.add(gm);

    threeR.current = {
      renderer,
      scene,
      camera,
      controls,
      mat,
      debugMat,
      mesh,
      G,
      light,
    };

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
      controls.update();
      renderer.render(scene, camera);
    };
    tick();

    return () => {
      cancelAnimationFrame(raf);
      renderer.dispose();
      ro.disconnect();
      controls.dispose();
    };
  }, []);

  useEffect(() => {
    const { renderer, scene } = threeR.current;
    if (!renderer || !scene) return;
    const prev = scene.environment;
    const e = buildEnv(renderer, hdri);
    scene.environment = e;
    scene.background = e;
    prev?.dispose();
  }, [hdri]);
  useEffect(() => {
    const { mesh, G } = threeR.current;
    if (!mesh) return;
    mesh.geometry = G[shape] ?? G.sphere;
  }, [shape]);
  useEffect(() => {
    const { mat, debugMat } = threeR.current;
    if (!mat) return;
    [
      mat.map,
      mat.normalMap,
      mat.roughnessMap,
      mat.metalnessMap,
      mat.aoMap,
      mat.displacementMap,
      debugMat.map,
    ].forEach((t) => {
      if (!t) return;
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.repeat.set(tileRep, tileRep);
      t.needsUpdate = true;
    });
  }, [tileRep]);

  useEffect(() => {
    const { scene, light, mat } = threeR.current;
    if (scene) scene.environmentRotation.y = settings.envRotation;
    if (light) {
      const r = 5;
      light.position.set(
        Math.cos(settings.envRotation) * r,
        5,
        Math.sin(settings.envRotation) * r,
      );
    }
    if (mat) mat.displacementScale = settings.dispScale;
  }, [settings.envRotation, settings.dispScale]);

  useEffect(() => {
    const { mesh, mat, debugMat } = threeR.current;
    if (!mesh || !maps) return;
    if (viewMode === "pbr") {
      mesh.material = mat;
    } else {
      mesh.material = debugMat;
      const tMap = {
        albedo: maps.albedo,
        normal: maps.normal,
        roughness: maps.roughness,
        metallic: maps.metallic,
      }[viewMode];
      if (tMap) {
        new THREE.TextureLoader().load(tMap, (t) => {
          t.wrapS = t.wrapT = THREE.RepeatWrapping;
          t.repeat.set(tileRep, tileRep);
          if (viewMode === "albedo") t.colorSpace = THREE.SRGBColorSpace;
          debugMat.map?.dispose();
          debugMat.map = t;
          debugMat.needsUpdate = true;
        });
      }
    }
  }, [viewMode, maps, tileRep]);

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
    if (maps.albedo) {
      ld(maps.albedo, (t) => {
        t.colorSpace = THREE.SRGBColorSpace;
        mat.map?.dispose();
        mat.map = t;
        mat.needsUpdate = true;
      });
    }
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
    ld(maps.displacement, (t) => {
      mat.displacementMap?.dispose();
      mat.displacementMap = t;
      mat.displacementScale = settings.dispScale;
      mat.needsUpdate = true;
    });
  }, [maps]);

  const ss = (k) => (e) =>
    setSettings((s) => ({ ...s, [k]: parseFloat(e.target.value) }));
  const tg = (k) => () => setSettings((s) => ({ ...s, [k]: !s[k] }));
  const handleRandomSeed = () =>
    setSettings((s) => ({ ...s, seed: randomSeed() }));

  const prefix = () => imgInfo?.name || "unarrived";
  const dl = (k) => {
    if (!maps?.[k]) return;
    const a = document.createElement("a");
    a.href = maps[k];
    a.download = `${prefix()}_${k}.png`;
    a.click();
  };

  const handleZip = async () => {
    if (!maps || exporting) return;
    setExporting(true);
    try {
      await doZip(maps, res, prefix());
    } finally {
      setExporting(false);
    }
  };
  const handleGLB = async () => {
    if (!maps || exporting) return;
    setExporting(true);
    try {
      const exporter = new GLTFExporter();
      const { mesh } = threeR.current;
      const exportMesh = mesh.clone();
      exportMesh.geometry = mesh.geometry.clone();
      exportMesh.material = mesh.material.clone();
      exportMesh.position.set(0, 0, 0);
      exportMesh.rotation.set(0, 0, 0);
      exporter.parse(
        exportMesh,
        (glb) => {
          const blob = new Blob([glb], { type: "application/octet-stream" });
          const a = document.createElement("a");
          a.href = URL.createObjectURL(blob);
          a.download = `${prefix()}_PBR.glb`;
          a.click();
          URL.revokeObjectURL(a.href);
          setExporting(false);
        },
        (err) => {
          console.error("[GLTFExporter]", err);
          setExporting(false);
          alert("Failed to generate GLB.");
        },
        { binary: true },
      );
    } catch (e) {
      console.error(e);
      setExporting(false);
    }
  };

  const step = STEPS[aiStep];

  return (
    <>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}html,body{height:100%;overflow:hidden;background:${C.bg}}body{font-family:'Inter',sans-serif;font-size:13px;color:${C.text};-webkit-font-smoothing:antialiased}::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:${C.sidebar}}::-webkit-scrollbar-thumb{background:${C.border};border-radius:4px}::-webkit-scrollbar-thumb:hover{background:${C.borderHi}}input[type=range]{-webkit-appearance:none;width:100%;height:3px;background:${C.panel};outline:none;cursor:pointer;border-radius:2px}input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:${C.accent};cursor:pointer;border:2px solid ${C.bg};box-shadow:0 0 0 1px ${C.accentDim}}input[type=range]:hover{background:${C.border}}@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}@keyframes spin{to{transform:rotate(360deg)}}@keyframes fadein{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}button{font-family:'Inter',sans-serif;transition:all .15s}button:hover:not(:disabled){filter:brightness(1.15)} select{appearance:none; background:${C.panel}; border:1px solid ${C.border}; color:${C.text}; padding:6px 10px; border-radius:4px; font-size:12px; width:100%; outline:none; cursor:pointer; font-family:'Inter',sans-serif;} select:focus{border-color:${C.accent}}`}</style>

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
            width: 320,
            flexShrink: 0,
            borderRight: `1px solid ${C.border}`,
            display: "flex",
            flexDirection: "column",
            background: C.sidebar,
            overflowY: "auto",
            overflowX: "hidden",
          }}
        >
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
              MAPPER{" "}
              <span
                style={{ fontSize: 12, color: C.pink, verticalAlign: "super" }}
              >
                v10 MAX
              </span>
            </div>
            <div
              style={{
                fontSize: 11,
                color: C.textDim,
                marginTop: 6,
                fontFamily: "'JetBrains Mono',monospace",
              }}
            >
              Procedural Material Engine
            </div>
          </div>

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
                    Drop Base Image
                  </div>
                </div>
              )}
            </div>
          </div>

          <div
            style={{
              padding: "8px 14px",
              borderBottom: `1px solid ${C.border}`,
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
                }}
              >
                {tileProg ? `Tile ${tileProg.c}/${tileProg.t}` : step.label}
              </div>
            </div>
            {tileProg && (
              <div
                style={{
                  marginTop: 6,
                  height: 3,
                  background: C.panel,
                  borderRadius: 2,
                }}
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
            )}
          </div>

          {/* 🔥 SEED GENERATOR */}
          <div
            style={{
              padding: "12px 14px",
              borderBottom: `1px solid ${C.border}`,
            }}
          >
            <Label>🎲 Procedural Generator</Label>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                background: C.panel,
                padding: "8px 10px",
                borderRadius: 6,
                border: `1px solid ${C.border}`,
                marginTop: 8,
              }}
            >
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 11, color: C.textDim }}>
                  Random Seed
                </div>
                <div
                  style={{
                    fontSize: 13,
                    fontWeight: 600,
                    fontFamily: "'JetBrains Mono',monospace",
                    color: C.accent,
                  }}
                >
                  {settings.seed.toFixed(1)}
                </div>
              </div>
              <button
                onClick={handleRandomSeed}
                style={{
                  background: C.accent,
                  color: "#000",
                  border: "none",
                  padding: "6px 12px",
                  borderRadius: 4,
                  fontWeight: 600,
                  cursor: "pointer",
                }}
              >
                Regenerate
              </button>
            </div>
          </div>

          <div style={{ padding: "8px 14px 14px" }}>
            <Section label="🎨 Albedo Tweaks">
              <Toggle
                label="HQ Seamless (FBM Organic)"
                value={settings.seamless}
                onChange={tg("seamless")}
                color={C.green}
              />
              <Slider
                label="Brightness"
                v={settings.brightness}
                min={0}
                max={2}
                step={0.05}
                onChange={ss("brightness")}
              />
              <Slider
                label="Contrast"
                v={settings.albContrast}
                min={0}
                max={2}
                step={0.05}
                onChange={ss("albContrast")}
              />
              <Slider
                label="Saturation"
                v={settings.saturation}
                min={0}
                max={2}
                step={0.05}
                onChange={ss("saturation")}
              />
            </Section>

            {/* 🔥 OLD METAL MODE */}
            <Section label="🛢️ Old Metal Mode (Rust & Scratches)">
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "8px 12px",
                  background: settings.metalMode ? "#3a2010" : C.panel,
                  borderRadius: 6,
                  border: `1px solid ${settings.metalMode ? "#a05020" : C.border}`,
                  marginBottom: 12,
                }}
              >
                <div
                  style={{
                    fontSize: 12,
                    fontWeight: 600,
                    color: settings.metalMode ? "#e8a020" : C.text,
                  }}
                >
                  {settings.metalMode ? "Old Metal Active" : "Generic Mode"}
                </div>
                <Toggle
                  value={settings.metalMode}
                  onChange={tg("metalMode")}
                  color="#e8a020"
                />
              </div>

              {settings.metalMode && (
                <div
                  style={{
                    padding: "10px",
                    background: "#2a1508",
                    borderRadius: 6,
                    border: "1px solid #603010",
                    marginBottom: 12,
                  }}
                >
                  <div
                    style={{
                      fontSize: 11,
                      fontWeight: 600,
                      color: "#e8a020",
                      marginBottom: 10,
                    }}
                  >
                    RUST PITTING
                  </div>
                  <Slider
                    label="Rust Coverage"
                    v={settings.rustLvl}
                    min={0}
                    max={1}
                    step={0.01}
                    onChange={ss("rustLvl")}
                  />
                  <Slider
                    label="Rust Scale"
                    v={settings.rustScale}
                    min={5}
                    max={50}
                    step={1}
                    onChange={ss("rustScale")}
                  />
                  <Slider
                    label="Rust Depth (Normal)"
                    v={settings.rustDepth}
                    min={0}
                    max={5}
                    step={0.1}
                    onChange={ss("rustDepth")}
                  />
                  <div
                    style={{
                      height: 1,
                      background: "#402010",
                      margin: "10px 0",
                    }}
                  />
                  <div
                    style={{
                      fontSize: 11,
                      fontWeight: 600,
                      color: "#e8a020",
                      marginBottom: 10,
                    }}
                  >
                    SCRATCHES
                  </div>
                  <Slider
                    label="Scratch Density"
                    v={settings.scratchLvl}
                    min={0}
                    max={1}
                    step={0.01}
                    onChange={ss("scratchLvl")}
                  />
                  <Slider
                    label="Scratch Length"
                    v={settings.scratchScale}
                    min={10}
                    max={150}
                    step={5}
                    onChange={ss("scratchScale")}
                  />
                  <Slider
                    label="Scratch Depth (Normal)"
                    v={settings.scratchDepth}
                    min={0}
                    max={3}
                    step={0.1}
                    onChange={ss("scratchDepth")}
                  />
                </div>
              )}
            </Section>

            <Section label="🌟 Smart Materials (Generic)">
              <Slider
                label="Edge Wear Level"
                v={settings.edgeWear}
                min={0}
                max={1}
                step={0.01}
                onChange={ss("edgeWear")}
              />
              <Slider
                label="Edge Wear Contrast"
                v={settings.edgeContrast}
                min={0.5}
                max={5}
                step={0.1}
                onChange={ss("edgeContrast")}
              />
              <Slider
                label="Grunge Amount"
                v={settings.grungeLvl}
                min={0}
                max={1}
                step={0.01}
                onChange={ss("grungeLvl")}
              />
              <Slider
                label="Micro FBM Noise"
                v={settings.microDetail}
                min={0}
                max={1}
                step={0.01}
                onChange={ss("microDetail")}
                hint="Procedural bumpy noise applied to surface"
              />
            </Section>

            <Section label="🔬 Micro-Surface (Substance Ready)">
              <div style={{ marginBottom: 10 }}>
                <select
                  value={settings.microType}
                  onChange={(e) =>
                    setSettings((s) => ({
                      ...s,
                      microType: parseFloat(e.target.value),
                    }))
                  }
                >
                  <option value={0}>None (Smooth)</option>
                  <option value={1}>Fine Grain / Sand</option>
                  <option value={2}>Leather / Skin Pores</option>
                  <option value={3}>Fabric / Canvas Weave</option>
                  <option value={4}>Brushed Metal</option>
                </select>
              </div>
              {settings.microType > 0 && (
                <>
                  <Slider
                    label="Micro Density (Scale)"
                    v={settings.microScale}
                    min={10}
                    max={1500}
                    step={10}
                    onChange={ss("microScale")}
                    hint="Higher means tinier details"
                  />
                  <Slider
                    label="Micro Depth"
                    v={settings.microDepth}
                    min={0}
                    max={2}
                    step={0.01}
                    onChange={ss("microDepth")}
                    hint="Intensity injected into Normal & Roughness"
                  />
                </>
              )}
            </Section>

            <Section label="🔮 Normal Map Setup">
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
              />
              <Slider
                label="Bilateral σ"
                v={settings.normalSigma}
                min={0.01}
                max={0.3}
                step={0.01}
                onChange={ss("normalSigma")}
                hint="Depth map blur/smoothing"
              />
              <Toggle
                label="DirectX (Invert Y)"
                value={settings.normalInvertY}
                onChange={tg("normalInvertY")}
              />
            </Section>

            <Section label="🔮 Roughness Setup">
              <Slider
                label="Contrast"
                v={settings.roughCon}
                min={0.5}
                max={6}
                step={0.1}
                onChange={ss("roughCon")}
              />
              <Slider
                label="Material Blend"
                v={settings.matBlend}
                min={0}
                max={1}
                step={0.01}
                onChange={ss("matBlend")}
              />
              <Toggle
                label="Invert Roughness"
                value={settings.roughInv}
                onChange={tg("roughInv")}
              />
            </Section>

            <Section label="🔮 Metallic Setup">
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
          <div
            style={{
              height: 44,
              borderBottom: `1px solid ${C.border}`,
              display: "flex",
              alignItems: "center",
              padding: "0 20px",
              gap: 16,
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
              <span style={{ fontSize: 12, color: C.textDim, fontWeight: 500 }}>
                {tileProg ? `Tile ${tileProg.c}/${tileProg.t}` : step.label}
              </span>
            </div>
            {maps?.albedo && (
              <span
                style={{
                  fontSize: 11,
                  background: C.green + "22",
                  color: C.green,
                  padding: "2px 8px",
                  borderRadius: 4,
                }}
              >
                ✓ Maps Generated
              </span>
            )}
            <div
              style={{
                marginLeft: "auto",
                fontSize: 11,
                color: C.textFaint,
                fontFamily: "'JetBrains Mono',monospace",
              }}
            >
              Infinite Seed · Deep Pitting · Full Parameters Restored
            </div>
          </div>

          <div
            style={{
              flex: 1,
              display: "grid",
              gridTemplateColumns: "repeat(3,1fr)",
              gridAutoRows: "minmax(180px,1fr)",
              gap: 2,
              padding: 2,
              overflowY: "auto",
              minHeight: 0,
            }}
          >
            {ALL_MAPS.map((m) => {
              if (maps && maps[m.key] === undefined) return null;
              return (
                <MapTile
                  key={m.key}
                  meta={m}
                  url={maps?.[m.key]}
                  hovered={hovered === m.key}
                  onEnter={() => setHovered(m.key)}
                  onLeave={() => setHovered(null)}
                  onDl={() => dl(m.key)}
                />
              );
            })}
          </div>
        </main>

        {/* ═══ RIGHT: PRO 3D PREVIEW ═══ */}
        <aside
          style={{
            width: 340,
            flexShrink: 0,
            borderLeft: `1px solid ${C.border}`,
            display: "flex",
            flexDirection: "column",
            background: C.sidebar,
            overflowY: "auto",
            overflowX: "hidden",
          }}
        >
          <div
            style={{
              height: 44,
              borderBottom: `1px solid ${C.border}`,
              display: "flex",
              alignItems: "center",
              padding: "0 16px",
              flexShrink: 0,
              justifyContent: "space-between",
            }}
          >
            <span style={{ fontSize: 13, fontWeight: 600, color: C.text }}>
              Pro 3D Viewer
            </span>
            <div style={{ fontSize: 10, color: C.textDim }}>
              Scroll: Zoom | R-Click: Pan
            </div>
          </div>

          <div
            style={{
              width: "100%",
              aspectRatio: "1/1",
              position: "relative",
              flexShrink: 0,
              background: "#000",
              borderBottom: `1px solid ${C.border}`,
            }}
          >
            <canvas
              ref={cvs}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: "100%",
                display: "block",
                outline: "none",
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

          <div
            style={{
              padding: "12px",
              borderBottom: `1px solid ${C.border}`,
              background: C.panel,
            }}
          >
            <Label>View Mode (Channel Inspector)</Label>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 4,
                marginTop: 8,
              }}
            >
              {[
                { k: "pbr", l: "PBR Render" },
                { k: "albedo", l: "Base Color" },
                { k: "normal", l: "Normal" },
                { k: "roughness", l: "Roughness" },
                { k: "metallic", l: "Metallic" },
              ].map((v) => (
                <button
                  key={v.k}
                  onClick={() => setViewMode(v.k)}
                  style={{
                    padding: "6px 0",
                    borderRadius: 4,
                    fontSize: 11,
                    background:
                      viewMode === v.k ? C.blue + "22" : "transparent",
                    border: `1px solid ${viewMode === v.k ? C.blue : C.border}`,
                    color: viewMode === v.k ? C.blue : C.textDim,
                  }}
                >
                  {v.l}
                </button>
              ))}
            </div>
          </div>

          <div
            style={{
              padding: "10px 12px",
              borderBottom: `1px solid ${C.border}`,
              flexShrink: 0,
            }}
          >
            <Label>Mesh & Tiling</Label>
            <div style={{ display: "flex", gap: 4, marginTop: 6 }}>
              {SHAPES.map((s) => (
                <button
                  key={s.k}
                  onClick={() => setShape(s.k)}
                  style={{
                    flex: 1,
                    padding: "5px 0",
                    borderRadius: 4,
                    fontSize: 11,
                    background: shape === s.k ? C.accent + "22" : "transparent",
                    border: `1px solid ${shape === s.k ? C.accent : C.border}`,
                    color: shape === s.k ? C.accent : C.textDim,
                  }}
                >
                  {s.l}
                </button>
              ))}
              {[1, 2, 4].map((n) => (
                <button
                  key={n}
                  onClick={() => setTileRep(n)}
                  style={{
                    flex: 1,
                    padding: "5px 0",
                    borderRadius: 4,
                    fontSize: 11,
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

          <div style={{ padding: "10px 12px", flexShrink: 0 }}>
            <Label>Environment (Shadows Active)</Label>
            <div
              style={{
                display: "flex",
                gap: 4,
                marginTop: 6,
                marginBottom: 12,
              }}
            >
              {Object.keys(HDRI).map((k) => (
                <button
                  key={k}
                  onClick={() => setHdri(k)}
                  style={{
                    flex: 1,
                    padding: "5px 0",
                    borderRadius: 4,
                    fontSize: 11,
                    background: hdri === k ? C.teal + "22" : "transparent",
                    border: `1px solid ${hdri === k ? C.teal : C.border}`,
                    color: hdri === k ? C.teal : C.textDim,
                  }}
                >
                  {k}
                </button>
              ))}
            </div>
            <Slider
              label="Light Rotation"
              v={settings.envRotation}
              min={0}
              max={Math.PI * 2}
              step={0.1}
              onChange={ss("envRotation")}
            />
            <Slider
              label="Displacement"
              v={settings.dispScale}
              min={0}
              max={0.5}
              step={0.01}
              onChange={ss("dispScale")}
            />
          </div>

          <div
            style={{
              padding: "16px 14px 20px",
              borderTop: `1px solid ${C.border}`,
              flexShrink: 0,
              marginTop: "auto",
            }}
          >
            <Label>Resolution & Export</Label>
            <div
              style={{
                display: "flex",
                gap: 4,
                marginTop: 6,
                marginBottom: 12,
              }}
            >
              {ALL_RES.map((r) => {
                const avail = r <= gpu.maxTex,
                  active = r === res;
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
                      background: active ? C.accent + "22" : "transparent",
                      border: `1px solid ${active ? C.accent : avail ? C.border : C.border + "44"}`,
                      color: active ? C.accent : avail ? C.text : C.textFaint,
                    }}
                  >
                    {r >= 1000 ? `${r / 1000}K` : r}
                  </button>
                );
              })}
            </div>
            <button
              onClick={handleZip}
              disabled={!maps || exporting}
              style={{
                width: "100%",
                padding: "10px 12px",
                borderRadius: 6,
                fontSize: 12,
                fontWeight: 600,
                background: maps && !exporting ? C.blue + "22" : "transparent",
                border: `1px solid ${maps && !exporting ? C.blue : C.border}`,
                color: maps && !exporting ? C.blue : C.textFaint,
                cursor: maps && !exporting ? "pointer" : "not-allowed",
                textAlign: "left",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 8,
              }}
            >
              <span>📦 Download All Maps (ZIP)</span>
              {exporting ? (
                <span style={{ animation: "spin 1s linear infinite" }}>⟳</span>
              ) : (
                <span>↓</span>
              )}
            </button>
            <button
              onClick={handleGLB}
              disabled={!maps || exporting}
              style={{
                width: "100%",
                padding: "10px 12px",
                borderRadius: 6,
                fontSize: 12,
                fontWeight: 600,
                background:
                  maps && !exporting ? C.accent + "22" : "transparent",
                border: `1px solid ${maps && !exporting ? C.accent : C.border}`,
                color: maps && !exporting ? C.accent : C.textFaint,
                cursor: maps && !exporting ? "pointer" : "not-allowed",
                textAlign: "left",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <span>🧊 Export 3D Model (.glb)</span>
              <span>↓</span>
            </button>
          </div>
        </aside>
      </div>
    </>
  );
}

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
        </div>
      )}
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
            fontSize: 11,
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
