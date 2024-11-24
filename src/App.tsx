import './App.css'
import Gpt from './my-gpt/Gpt'
import ToxicityDetector from "../src/Toxicity/Toxicity.tsx";
import Transformers from "../src/Transformers/Transformers.tsx";

function App() {

  return (
    <>
       {/*<ToxicityDetector/>*/}
       <Transformers/>
      {/*<Gpt/>*/}
    </>
  )
}

export default App
