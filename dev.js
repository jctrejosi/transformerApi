#!/usr/bin/env node
/**
 * transformerApi — launcher del servicio.
 *
 * Crea el entorno virtual (venv/) si no existe, instala las dependencias de
 * requirements.txt la primera vez y arranca el servidor con uvicorn.
 *
 * Uso:
 *   node dev.js            → crea venv si falta, instala deps (1ª vez) y arranca
 *   node dev.js --install  → reinstala requirements.txt aunque ya exista venv
 *   node dev.js --reload   → arranca con --reload (recarga en desarrollo)
 *   node dev.js --port 9000→ cambia el puerto (por defecto 8000)
 *   node dev.js --host 127.0.0.1 → cambia el host (por defecto 0.0.0.0)
 *   node dev.js --help     → esta ayuda
 *
 * Si existe un archivo .env en la raíz, se pasa a uvicorn con --env-file.
 */

const { spawn, spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const ROOT = __dirname;
const VENV_DIR = path.join(ROOT, "venv");
const REQS_FILE = path.join(ROOT, "requirements.txt");
const ENV_FILE = path.join(ROOT, ".env");

const IS_WIN = process.platform === "win32";

const PYTHON_BIN = IS_WIN
  ? path.join(VENV_DIR, "Scripts", "python.exe")
  : path.join(VENV_DIR, "bin", "python");
const PIP_BIN = IS_WIN
  ? path.join(VENV_DIR, "Scripts", "pip.exe")
  : path.join(VENV_DIR, "bin", "pip");

// ── Argumentos ──────────────────────────────────────────────

const args = process.argv.slice(2);
const HELP = args.includes("--help") || args.includes("-h");
const INSTALL = args.includes("--install") || args.includes("-i");
const RELOAD = args.includes("--reload");

function readArg(flag) {
  const i = args.findIndex((a) => a === flag);
  return i >= 0 && args[i + 1] ? args[i + 1] : null;
}

const PORT = readArg("--port") || readArg("-p") || "8000";
const HOST = readArg("--host") || "0.0.0.0";

// ── Utilidades ──────────────────────────────────────────────

function log(msg) {
  console.log(`\x1b[36m[dev]\x1b[0m ${msg}`);
}

function ok(msg) {
  console.log(`\x1b[32m[dev]\x1b[0m ✓ ${msg}`);
}

function fail(msg) {
  console.error(`\x1b[31m[dev]\x1b[0m ✘ ${msg}`);
}

/** Encuentra un intérprete de Python disponible en el sistema. */
function findSystemPython() {
  const candidates = IS_WIN ? ["python", "py"] : ["python3", "python"];
  for (const c of candidates) {
    const r = spawnSync(c, ["--version"], { stdio: "ignore" });
    if (r.status === 0) return c;
  }
  return null;
}

function runSync(cmd, cmdArgs, opts = {}) {
  const r = spawnSync(cmd, cmdArgs, { stdio: "inherit", ...opts });
  if (r.error) {
    fail(`no se pudo ejecutar: ${cmd} (${r.error.message})`);
    return 1;
  }
  return r.status === null ? 1 : r.status;
}

// ── Entorno virtual ─────────────────────────────────────────

function venvReady() {
  return (
    fs.existsSync(PYTHON_BIN) &&
    fs.existsSync(PIP_BIN) &&
    fs.existsSync(REQS_FILE)
  );
}

function ensureVenv() {
  if (venvReady() && !INSTALL) {
    ok(`entorno virtual detectado (${path.basename(VENV_DIR)}/)`);
    return true;
  }

  const python = findSystemPython();
  if (!python) {
    fail("no se encontró Python en el PATH (probé: python3 / python).");
    return false;
  }

  if (!fs.existsSync(PYTHON_BIN)) {
    log(`creando entorno virtual con ${python}...`);
    if (runSync(python, ["-m", "venv", VENV_DIR]) !== 0) {
      fail("falló la creación del entorno virtual.");
      return false;
    }
    ok("entorno virtual creado.");
  }

  log("actualizando pip...");
  if (runSync(PYTHON_BIN, ["-m", "pip", "install", "--upgrade", "pip"]) !== 0) {
    fail("no se pudo actualizar pip.");
    return false;
  }

  log("instalando dependencias (requirements.txt)... esto puede tardar con torch.");
  if (runSync(PIP_BIN, ["install", "-r", REQS_FILE]) !== 0) {
    fail("falló la instalación de dependencias.");
    return false;
  }
  ok("dependencias instaladas.");

  return true;
}

// ── Arranque ────────────────────────────────────────────────

function startServer() {
  const uvicornArgs = [
    "-m",
    "uvicorn",
    "apis.main:app",
    "--host",
    HOST,
    "--port",
    PORT,
  ];

  if (RELOAD) uvicornArgs.push("--reload");
  if (fs.existsSync(ENV_FILE)) {
    uvicornArgs.push("--env-file", ENV_FILE);
    log("usando variables de entorno desde .env");
  }

  log(`arrancando en http://${HOST}:${PORT}  (docs: /docs)`);
  const child = spawn(PYTHON_BIN, uvicornArgs, {
    cwd: ROOT,
    stdio: "inherit",
    shell: IS_WIN,
  });

  child.on("error", (err) => {
    fail(`no se pudo arrancar el servidor: ${err.message}`);
    process.exit(1);
  });

  child.on("exit", (code) => {
    process.exit(code ?? 0);
  });
}

// ── Main ────────────────────────────────────────────────────

function main() {
  if (HELP) {
    console.log(
      "transformerApi launcher\n" +
        "  node dev.js            → crea venv si falta, instala deps (1ª vez) y arranca\n" +
        "  node dev.js --install  → reinstala requirements.txt\n" +
        "  node dev.js --reload   → arranca con --reload\n" +
        "  node dev.js --port 9000→ cambia el puerto (default 8000)\n" +
        "  node dev.js --host 127.0.0.1 → cambia el host (default 0.0.0.0)\n",
    );
    return;
  }

  if (!ensureVenv()) {
    process.exit(1);
  }

  startServer();
}

main();
