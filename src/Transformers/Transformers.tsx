import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { pipeline } from "@xenova/transformers";

type WorkerMessage = {
  status: 'initiate' | 'ready' | 'complete';
  output?: any[]; // Adjust the type of `output` as needed based on your expected data structure
};

const task = 'text-classification';
const model = 'Xenova/distilbert-base-uncased-finetuned-sst-2-english';

const classifyBlocking = await pipeline(task, model);

export default function Transformers() {
  // Keep track of the classification result and the model loading status.
  const [result, setResult] = useState<any>(null); // Use a more specific type if possible
  const [ready, setReady] = useState<boolean | null>(null);


  // Create a reference to the worker object.
  const worker = useRef<Worker | null>(null);
  

  // We use the `useEffect` hook to set up the worker as soon as the `App` component is mounted.
  useEffect(() => {
    if (!worker.current) {
      // Create the worker if it does not yet exist.
      worker.current = new Worker(new URL('./worker.js', import.meta.url), {
        type: 'module',
      });
    }

    // Create a callback function for messages from the worker thread.
    const onMessageReceived = (e: MessageEvent<WorkerMessage>) => {
      switch (e.data.status) {
        case 'initiate':
          setReady(false);
          break;
        case 'ready':
          setReady(true);
          break;
        case 'complete':
          console.log(e.data.output);
          setResult(e.data.output ? e.data.output[0] : null);
          break;
      }
    };

    // Attach the callback function as an event listener.
    worker.current.addEventListener('message', onMessageReceived);

    // Define a cleanup function for when the component is unmounted.
    return () => worker.current?.removeEventListener('message', onMessageReceived);
  }, []);

  const classify = useCallback((text: string) => {
    if (worker.current) {
      worker.current.postMessage({ text });
    }
  }, []);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-12">
      <h1 className="text-5xl font-bold mb-2 text-center">Transformers.js</h1>
      <input
        type="text"
        className="w-full max-w-xs p-2 border border-gray-300 rounded mb-4"
        placeholder="Enter text here"
        // onInput={(e) => classify((e.target as HTMLInputElement).value)}
        onInput={async (e) => {
          const res = await classifyBlocking((e.target as HTMLInputElement).value)
          console.log(res);
        }}
      />

      {ready !== null && (
        <pre className="p-2 rounded">
          {!ready || !result ? 'Loading...' : JSON.stringify(result, null, 2)}
        </pre>
      )}
    </main>
  );
}
