import { TrainableGRUMLP } from "../trainable.js";
import { mseGradient, meanSquaredError } from "./loss.js";
import { Optimizer } from "./optimizer.js";
import { clipGradients, globalGradientNorm } from "./gradient-clipping.js";

export interface TrainableSample { sequence:number[][]; staticFeatures:number[]; target:number[]; }
export interface TrainableTrainerConfig { epochs?:number; clipNorm?:number; }
export interface TrainHistory { loss:number; gradientNorm:number; }

export class TrainableTrainer {
  constructor(private readonly optimizer:Optimizer, private readonly config:TrainableTrainerConfig={}) {}

  train(model:TrainableGRUMLP,data:TrainableSample[]):TrainHistory[]{
    if(data.length===0) throw new Error('training dataset must not be empty');
    const epochs=this.config.epochs??1; if(!Number.isInteger(epochs)||epochs<=0) throw new Error('epochs must be positive');
    const history:TrainHistory[]=[];
    for(let epoch=0;epoch<epochs;epoch++){
      let total=0; let norm=0;
      for(const sample of data){
        const prediction=model.forward(sample.sequence,sample.staticFeatures);
        total+=meanSquaredError(prediction,sample.target);
        model.backward(mseGradient(prediction,sample.target));
        const refs=model.parameterRefs(); const grads=refs.map(r=>r.gradient); norm=Math.max(norm,globalGradientNorm(grads));
        if(this.config.clipNorm!==undefined) clipGradients(grads,this.config.clipNorm);
        this.optimizer.step(refs); model.zeroGradients();
      }
      history.push({loss:total/data.length,gradientNorm:norm});
    }
    return history;
  }

  reset():void{this.optimizer.reset();}
}
