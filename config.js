/* Deployment settings.
 *
 * INTERPRETER_URL: the Cloudflare Worker that reads "Describe a look" with
 * Claude, e.g. 'https://facecam-interpreter.<account>.workers.dev'. Leave it
 * empty and the app reads the text on the device instead, with a fixed word
 * list. See DEPLOY.md, step 3.
 */
export const INTERPRETER_URL = '';
