import { Vector, assertVector, sigmoid, tanh } from "./core/tensor.js";
import { ParameterRef } from "./training/optimizer.js";

export interface TrainableGRUMLPConfig {
  sequenceInputSize: number;
  hiddenSize: number;
  staticInputSize: number;
  headHiddenSizes: number[];
  outputSize: number;
}

export interface BackwardResult { inputGradient: Vector[]; staticGradient: Vector; lossGradient: Vector; }

function zeros(n: number): number[] { return new Array(n).fill(0); }
function matrix(rows: number, cols: number): number[][] { return Array.from({length: rows}, () => zeros(cols)); }
function init(rows: number, cols: number): number[][] { const s=Math.sqrt(1/cols); return Array.from({length:rows},()=>Array.from({length:cols},()=>((Math.random()*2-1)*s))); }
function add(a:number[],b:number[]):number[]{ return a.map((v,i)=>v+b[i]); }
function outer(a:number[],b:number[]):number[][] { return a.map(x=>b.map(y=>x*y)); }
function matVec(m:number[][],v:number[]):number[]{ return m.map(r=>r.reduce((s,x,i)=>s+x*v[i],0)); }
function transposeVec(m:number[][],g:number[]):number[]{ const out=zeros(m[0].length); for(let i=0;i<m.length;i++)for(let j=0;j<m[i].length;j++)out[j]+=m[i][j]*g[i]; return out; }
function act(x:number,a:string):number { if(a==='relu') return Math.max(0,x); if(a==='tanh') return Math.tanh(x); return x; }
function deriv(y:number,a:string):number { if(a==='relu') return y>0?1:0; if(a==='tanh') return 1-y*y; return 1; }

export class TrainableGRUMLP {
  readonly config: TrainableGRUMLPConfig;
  readonly wz:number[][]; readonly wr:number[][]; readonly wh:number[][];
  readonly bz:Vector; readonly br:Vector; readonly bh:Vector;
  readonly denseWeights:number[][][]; readonly denseBiases:Vector[];
  readonly denseActivations:string[];

  private cache:{x:Vector[]; h:Vector[]; z:Vector[]; r:Vector[]; n:Vector[]} | null = null;
  private denseCache:Vector[] = [];
  private grads:ParameterRef[] = [];

  constructor(config:TrainableGRUMLPConfig){
    if(config.sequenceInputSize<=0||config.hiddenSize<=0||config.staticInputSize<0||config.outputSize<=0) throw new Error('invalid model dimensions');
    this.config=config; const n=config.sequenceInputSize+config.hiddenSize;
    this.wz=init(config.hiddenSize,n); this.wr=init(config.hiddenSize,n); this.wh=init(config.hiddenSize,n);
    this.bz=zeros(config.hiddenSize); this.br=zeros(config.hiddenSize); this.bh=zeros(config.hiddenSize);
    const sizes=[config.hiddenSize+config.staticInputSize,...config.headHiddenSizes,config.outputSize];
    this.denseWeights=[]; this.denseBiases=[]; this.denseActivations=[];
    for(let i=0;i<sizes.length-1;i++){ this.denseWeights.push(init(sizes[i+1],sizes[i])); this.denseBiases.push(zeros(sizes[i+1])); this.denseActivations.push(i===sizes.length-2?'linear':'relu'); }
  }

  forward(sequence:Vector[],staticFeatures:Vector):Vector{
    if(sequence.length===0) throw new Error('sequence must not be empty'); assertVector(staticFeatures,'staticFeatures');
    if(staticFeatures.length!==this.config.staticInputSize) throw new Error('static feature dimension mismatch');
    const h=[zeros(this.config.hiddenSize)],z:Vector[]=[],r:Vector[]=[],n:Vector[]=[];
    for(const x of sequence){ if(x.length!==this.config.sequenceInputSize) throw new Error('sequence input dimension mismatch'); const prev=h[h.length-1]; const c=[...x,...prev];
      const zz=this.wz.map((row,i)=>sigmoid(row.reduce((s,v,j)=>s+v*c[j],0)+this.bz[i]));
      const rr=this.wr.map((row,i)=>sigmoid(row.reduce((s,v,j)=>s+v*c[j],0)+this.br[i]));
      const ci=[...x,...prev.map((v,i)=>v*rr[i])]; const nn=this.wh.map((row,i)=>tanh(row.reduce((s,v,j)=>s+v*ci[j],0)+this.bh[i]));
      h.push(prev.map((v,i)=>(1-zz[i])*v+zz[i]*nn[i])); z.push(zz); r.push(rr); n.push(nn);
    }
    this.cache={x:sequence.map(v=>[...v]),h,z,r,n}; let value=[...h[h.length-1],...staticFeatures]; this.denseCache=[value];
    for(let l=0;l<this.denseWeights.length;l++){ value=this.denseWeights[l].map((row,i)=>act(row.reduce((s,w,j)=>s+w*value[j],0)+this.denseBiases[l][i],this.denseActivations[l])); this.denseCache.push(value); }
    return value;
  }

  backward(lossGradient:Vector):BackwardResult{
    if(!this.cache) throw new Error('forward must be called before backward');
    const p=this.parameterRefs(); this.grads.forEach(g=>g.gradient.fill(0));
    let g=[...lossGradient];
    for(let l=this.denseWeights.length-1;l>=0;l--){ const out=this.denseCache[l+1], input=this.denseCache[l], pre=g.map((v,i)=>v*deriv(out[i],this.denseActivations[l]));
      for(let i=0;i<this.denseWeights[l].length;i++){ this.denseBiases[l][i]; for(let j=0;j<input.length;j++) this.grads.find(q=>q.value===this.denseWeights[l][i])!.gradient[j]+=pre[i]*input[j]; }
      const bg=this.grads.find(q=>q.value===this.denseBiases[l])!.gradient; for(let i=0;i<pre.length;i++)bg[i]+=pre[i]; g=transposeVec(this.denseWeights[l],pre);
    }
    const hiddenG=g.slice(0,this.config.hiddenSize); const staticGradient=g.slice(this.config.hiddenSize);
    const dx=this.cache.x.map(()=>zeros(this.config.sequenceInputSize)); let gh=hiddenG;
    for(let t=this.cache.x.length-1;t>=0;t--){ const x=this.cache.x[t], prev=this.cache.h[t], zz=this.cache.z[t],rr=this.cache.r[t],nn=this.cache.n[t];
      const dprev=zeros(this.config.hiddenSize), dz=zeros(this.config.hiddenSize),dn=zeros(this.config.hiddenSize),dr=zeros(this.config.hiddenSize);
      for(let i=0;i<this.config.hiddenSize;i++){ dz[i]=gh[i]*(nn[i]-prev[i]); dn[i]=gh[i]*zz[i]; dprev[i]=gh[i]*(1-zz[i]); }
      const dcn=dn.map((v,i)=>v*(1-nn[i]*nn[i])); const dci=transposeVec(this.wh,dcn);
      for(let i=0;i<this.config.hiddenSize;i++)dr[i]=dci[this.config.sequenceInputSize+i]*prev[i];
      const drpre=dr.map((v,i)=>v*rr[i]*(1-rr[i])); const dzpre=dz.map((v,i)=>v*zz[i]*(1-zz[i]));
      const dcZ=transposeVec(this.wz,dzpre), dcR=transposeVec(this.wr,drpre);
      for(let i=0;i<this.config.hiddenSize;i++){ this.grads.find(q=>q.value===this.bz)!.gradient[i]+=dzpre[i]; this.grads.find(q=>q.value===this.br)!.gradient[i]+=drpre[i]; this.grads.find(q=>q.value===this.bh)!.gradient[i]+=dcn[i];
        for(let j=0;j<this.config.sequenceInputSize+this.config.hiddenSize;j++){ const c=[...x,...prev]; this.grads.find(q=>q.value===this.wz[i])!.gradient[j]+=dzpre[i]*c[j]; this.grads.find(q=>q.value===this.wr[i])!.gradient[j]+=drpre[i]*c[j]; const ci=[...x,...prev.map((v,k)=>v*rr[k])]; this.grads.find(q=>q.value===this.wh[i])!.gradient[j]+=dcn[i]*ci[j]; }
      }
      for(let j=0;j<this.config.sequenceInputSize;j++)dx[t][j]=dci[j]+dcZ[j]+dcR[j];
      for(let i=0;i<this.config.hiddenSize;i++)dprev[i]+=dcZ[this.config.sequenceInputSize+i]+dcR[this.config.sequenceInputSize+i]+dci[this.config.sequenceInputSize+i]*rr[i]; gh=dprev;
    }
    return {inputGradient:dx,staticGradient,lossGradient:[...lossGradient]};
  }

  parameterRefs():ParameterRef[]{
    if(this.grads.length) return this.grads;
    const add=(value:number[])=>{const ref={value,gradient:zeros(value.length)};this.grads.push(ref);return ref;};
    for(const m of [this.wz,this.wr,this.wh]) for(const row of m)add(row); add(this.bz);add(this.br);add(this.bh); for(const m of this.denseWeights)for(const row of m)add(row);for(const b of this.denseBiases)add(b); return this.grads;
  }

  zeroGradients():void{ this.parameterRefs().forEach(p=>p.gradient.fill(0)); }
}
