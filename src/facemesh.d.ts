declare module '@tensorflow-models/facemesh' {
  export function load(config?: { maxFaces?: number }): Promise<unknown>
}
