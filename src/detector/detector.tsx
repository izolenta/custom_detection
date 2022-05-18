import './detector.css';
import {ChangeEvent, ReactNode, useEffect, useRef, useState} from "react";
import * as tf from '@tensorflow/tfjs'

let model:tf.GraphModel;

const types = [
  {
    name: 'škoda',
    id: 1,
  },
  {
    name: 'volkswagen',
    id: 2,
  },
  {
    name: 'toyota',
    id: 3,
  },
];

interface Prediction {
  box: number[];
  type: string;
  score: number;
}

const Detector = () => {

  const [isInitialized, setInitialized] = useState(false);
  const [boundingData, setBoundingData] = useState(Array<Prediction>());
  const [boxes, setBoxes] = useState(Array<ReactNode>());

  useEffect(() => {
    tf.loadGraphModel('/data/model.json').then((result:tf.GraphModel) => {
      model = result;
      setInitialized(true);
    });
  }, [])

  useEffect(() => {
    let nodes = [];
    for (let next of boundingData) {
      let positionFrame = {left: next.box[0]+'px', top: next.box[1]+'px', width: next.box[2]+'px', height: next.box[3]+'px'};
      let positionHeader = {left: next.box[0]+'px', top: (next.box[1]-60)+'px'};
      let box = <div className={'positioned'} style={positionFrame}></div>;
      let header = <div className={'header'} style={positionHeader}>{next.type} {next.score.toFixed(2)}</div>;
      nodes.push(box);
      nodes.push(header);
    }
    setBoxes(nodes);
  }, [boundingData])

  const imageRef = useRef<HTMLImageElement>(null);

  async function fileLoaded(e: ChangeEvent<HTMLInputElement>) {
    setBoundingData([]);
    if (e.target.files) {
      let file = e.target.files[0];

      const image = imageRef.current!;
      let fr = new FileReader();

      fr.onload = function() {
        if (fr !== null && typeof fr.result == "string") {
          image.src = fr.result;
        }
      };
      fr.readAsDataURL(file);

      image.onload = async function() {
        let tensor = tf.browser.fromPixels(imageRef.current!).expandDims(0);//.reshape([1, imageRef.current!.width, imageRef.current!.height, 3]);
        tf.engine().startScope();
        let result = (await model.executeAsync(tensor, [ 'Identity_1:0','Identity_4:0','Identity_2:0']) as tf.Tensor[]);
        const boxes = result[0].arraySync() as number[][][];
        const scores = result[1].arraySync() as number[][];
        const classes = result[2].arraySync() as number[][];

        let detectedObjects:Prediction[] = [];

        scores[0].forEach((score, i) => {
          if (score > 0.7) {
            const bbox = [];
            const minY = boxes[0][i][0] * imageRef.current!.height;
            const minX = boxes[0][i][1] * imageRef.current!.width;
            const maxY = boxes[0][i][2] * imageRef.current!.height;
            const maxX = boxes[0][i][3] * imageRef.current!.width;
            bbox[0] = minX;
            bbox[1] = minY;
            bbox[2] = maxX - minX;
            bbox[3] = maxY - minY;
            detectedObjects.push({
              box: bbox,
              type: types.map((f) => f.name)[classes[0][i]-1],
              score: score,
             })
          }
        })

        console.log(result);
        setBoundingData(detectedObjects);
      };
    }
  }

  if (isInitialized) {
    return (
      <div>
        <div className={'headline'}>Custom Škoda, VW and Toyota Logo Detection model</div>
        <input className={'input'} type='file' accept={'image/*'} multiple={false} onChange={(e) => fileLoaded(e)}/>
        <div className={'wrapper'}>
          <img ref={imageRef} width={1000}></img>
          { boxes }
        </div>
      </div>
    );
  }
  else {
    return(
      <div>Loading...</div>
    );
  }
}

export default Detector;
