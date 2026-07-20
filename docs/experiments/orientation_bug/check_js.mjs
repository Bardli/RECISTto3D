// Ported helpers from app_three_models.py (verbatim logic)
function volPermRAS(vol) {
  const p = vol && vol.permRAS;
  if (Array.isArray(p) && p.length === 3 && p.every(v => Number.isFinite(v) && v !== 0)) return p.slice();
  return [1, 2, 3];
}
function rasVoxToNative(ras, dimsRAS, perm) {
  const nat = [0,0,0];
  for (let i=0;i<3;i++){ const ax=Math.abs(perm[i])-1; let v=ras[i]; if(perm[i]<0) v=(dimsRAS[i]-1)-v; nat[ax]=v; }
  return nat;
}
function nativeToRasVox(nat, dimsRAS, perm) {
  const ras=[0,0,0];
  for (let i=0;i<3;i++){ const ax=Math.abs(perm[i])-1; let v=nat[ax]; if(perm[i]<0) v=(dimsRAS[i]-1)-v; ras[i]=v; }
  return ras;
}
function nativeMaxForAxis(vol, nativeAxis){
  const perm=volPermRAS(vol); const dimsRAS=vol.dims||[];
  const dRAS=[dimsRAS[1]||1,dimsRAS[2]||1,dimsRAS[3]||1];
  for (let i=0;i<3;i++){ if(Math.abs(perm[i])-1===nativeAxis) return dRAS[i]-1; }
  return (dRAS[nativeAxis]||1)-1;
}

// Faithful NiiVue perm/flip from a 3x3 affine (port of calculateRAS)
function permFlip(a){
  const absR=a.map(r=>r.map(Math.abs));
  const ixyz=[1,1,1];
  if(absR[1][0]>absR[0][0]) ixyz[0]=2;
  if(absR[2][0]>absR[0][0] && absR[2][0]>absR[1][0]) ixyz[0]=3;
  if(ixyz[0]===1) ixyz[1]= absR[1][1]>absR[2][1]?2:3;
  else if(ixyz[0]===2) ixyz[1]= absR[0][1]>absR[2][1]?1:3;
  else ixyz[1]= absR[0][1]>absR[1][1]?1:2;
  ixyz[2]=6-ixyz[1]-ixyz[0];
  const perm=[1,2,3]; perm[ixyz[0]-1]=1; perm[ixyz[1]-1]=2; perm[ixyz[2]-1]=3;
  const R=[[0,0,0],[0,0,0],[0,0,0]];
  for(let i=0;i<3;i++)for(let j=0;j<3;j++) R[i][j]=a[i][perm[j]-1];
  const flip=[R[0][0]<0?1:0, R[1][1]<0?1:0, R[2][2]<0?1:0];
  return {perm,flip};
}
function permRAS(perm,flip){ const p=perm.slice(); for(let i=0;i<3;i++) if(flip[i]) p[i]=-p[i]; return p; }
function nativeToRas(natTruth, dimsNative, perm, flip){
  const ras=[0,0,0];
  for(let out=0;out<3;out++){ const src=perm[out]-1; let idx=natTruth[src]; if(flip[out]) idx=dimsNative[src]-1-idx; ras[out]=idx; }
  return ras;
}
function rasDims(dimsNative, perm){ return [dimsNative[perm[0]-1],dimsNative[perm[1]-1],dimsNative[perm[2]-1]]; }

// End-to-end with the FIX (mirrors app pointFromEvent)
function appFixed(clickNative, dimsNative, aff, isRad=true){
  const {perm,flip}=permFlip(aff);
  const dRAS=rasDims(dimsNative,perm);
  const pras=permRAS(perm,flip);
  const ras=nativeToRas(clickNative, dimsNative, perm, flip);      // NiiVue frac2vox
  // vol as the app sees it:
  const vol={ dims:[dimsNative.length, dRAS[0], dRAS[1], dRAS[2]], permRAS:pras };
  const nat=rasVoxToNative(ras, dRAS, volPermRAS(vol));
  const vx = isRad ? nativeMaxForAxis(vol,0)-nat[0] : nat[0];
  const vy = nativeMaxForAxis(vol,1)-nat[1];
  const vz = nat[2];
  return [vx,vy,vz];
}

const dims=[512,512,113];
const click=[100,380,40];
const cases={
  identity:[[1,0,0],[0,1,0],[0,0,1]],
  flippedXY:[[-1,0,0],[0,-1,0],[0,0,1]],
  swappedXY:[[0,1,0],[1,0,0],[0,0,1]],
  flipX:[[-1,0,0],[0,1,0],[0,0,1]],
};
let base=null, ok=true;
for(const [name,aff] of Object.entries(cases)){
  const got=appFixed(click,dims,aff);
  if(base===null) base=got;
  const same=JSON.stringify(got)===JSON.stringify(base);
  ok=ok&&same;
  console.log(name.padEnd(10), "->", JSON.stringify(got), same?"consistent":"DIVERGED");
}
console.log(ok?"\nJS FIX VERIFIED (all directions consistent)":"\nJS FIX FAILED");
process.exit(ok?0:1);
