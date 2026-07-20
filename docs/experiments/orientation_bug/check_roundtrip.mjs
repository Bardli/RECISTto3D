function volPermRAS(vol){const p=vol&&vol.permRAS;if(Array.isArray(p)&&p.length===3&&p.every(v=>Number.isFinite(v)&&v!==0))return p.slice();return[1,2,3];}
function rasVoxToNative(ras,dimsRAS,perm){const nat=[0,0,0];for(let i=0;i<3;i++){const ax=Math.abs(perm[i])-1;let v=ras[i];if(perm[i]<0)v=(dimsRAS[i]-1)-v;nat[ax]=v;}return nat;}
function nativeToRasVox(nat,dimsRAS,perm){const ras=[0,0,0];for(let i=0;i<3;i++){const ax=Math.abs(perm[i])-1;let v=nat[ax];if(perm[i]<0)v=(dimsRAS[i]-1)-v;ras[i]=v;}return ras;}
function nativeMaxForAxis(vol,nativeAxis){const perm=volPermRAS(vol);const dimsRAS=vol.dims||[];const dRAS=[dimsRAS[1]||1,dimsRAS[2]||1,dimsRAS[3]||1];for(let i=0;i<3;i++){if(Math.abs(perm[i])-1===nativeAxis)return dRAS[i]-1;}return(dRAS[nativeAxis]||1)-1;}

const isRadiological=true;
// forward: RAS vox (from frac2vox) -> model coord  (pointFromEvent tail)
function pointFwd(vol, rasVox){
  const dimsRAS=vol.dims||[];
  const perm=volPermRAS(vol);
  const nat=rasVoxToNative(rasVox,[dimsRAS[1],dimsRAS[2],dimsRAS[3]],perm);
  return [isRadiological?nativeMaxForAxis(vol,0)-nat[0]:nat[0], nativeMaxForAxis(vol,1)-nat[1], nat[2]];
}
// reverse: model coord -> RAS vox (modelVoxToRasVox) -> should feed vox2frac
function modelVoxToRasVox(vol,mx,my,mz){
  const perm=volPermRAS(vol);const dimsRAS=vol.dims||[];const dRAS=[dimsRAS[1]||1,dimsRAS[2]||1,dimsRAS[3]||1];
  const nx=isRadiological?nativeMaxForAxis(vol,0)-mx:mx;
  const ny=nativeMaxForAxis(vol,1)-my;
  const nz=mz;
  return nativeToRasVox([nx,ny,nz],dRAS,perm);
}

// Try several vols (permRAS variants) and random RAS voxels; forward then reverse must return original RAS vox.
const vols=[
  {dims:[3,512,512,113],permRAS:[1,2,3]},
  {dims:[3,512,512,113],permRAS:[-1,-2,3]},
  {dims:[3,512,512,113],permRAS:[2,1,3]},
  {dims:[3,512,512,113],permRAS:[-2,1,3]},
  {dims:[3,512,113,512],permRAS:[1,3,2]},
];
let ok=true;
for(const vol of vols){
  for(let t=0;t<200;t++){
    const ras=[Math.floor(Math.random()*vol.dims[1]),Math.floor(Math.random()*vol.dims[2]),Math.floor(Math.random()*vol.dims[3])];
    const model=pointFwd(vol,ras);
    const back=modelVoxToRasVox(vol,model[0],model[1],model[2]);
    if(JSON.stringify(back)!==JSON.stringify(ras)){ ok=false; console.log("MISMATCH",vol.permRAS,"ras",ras,"model",model,"back",back); break; }
  }
}
console.log(ok?"ROUND-TRIP EXACT for all permRAS variants (draw==redraw)":"ROUND-TRIP BROKEN");
process.exit(ok?0:1);
