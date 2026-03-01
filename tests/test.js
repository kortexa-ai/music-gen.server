#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const endpoint = process.env.MUSIC_ENDPOINT ?? 'http://127.0.0.1:4009/generate';

const payload = {
  caption: 'A gentle lo-fi hip hop beat with warm piano chords and vinyl crackle',
  lyrics: '[Instrumental]',
  instrumental: true,
  duration: 15,
  inference_steps: 8,
  guidance_scale: 7.0,
  seed: 42,
  batch_size: 1,
  audio_format: 'flac',
  thinking: false,
};

async function main() {
  console.log(`POST ${endpoint}`);
  console.log('Payload:', JSON.stringify(payload, null, 2));

  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Request failed with ${response.status}: ${text}`);
  }

  const result = await response.json();
  if (!result.audios?.length) {
    throw new Error('No audios returned in response');
  }

  const [firstAudio] = result.audios;
  const buffer = Buffer.from(firstAudio, 'base64');
  const outPath = path.join(__dirname, 'generated.flac');
  await fs.writeFile(outPath, buffer);
  console.log(`Saved audio to ${outPath} (${buffer.length} bytes)`);

  const metadata = result.metadata ?? {};
  console.log('Metadata:', metadata);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
