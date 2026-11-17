/*
 * Rosette: a braided 3D torus drawn in the style of the guilloche patterns
 * printed on banknotes. Pure canvas, no libraries, so it works offline.
 *
 * The pattern is driven by one number, `order` (0 to 1):
 *   1 = tight, regular weave      0 = loose, turbulent weave
 * Pages feed it real data (CIBIL score, approval probability).
 *
 * Usage:   <canvas data-rosette data-order="0.6" data-palette="gold"></canvas>
 * Control: canvas._rosette.set({ order: 0.9, palette: "mint" });
 *          canvas._rosette.burst();   // quick spin that settles
 */
(function () {
  "use strict";

  var TAU = Math.PI * 2;
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var PALETTES = {
    gold:  { far: [26, 112, 108],  near: [246, 190, 84]  },
    mint:  { far: [22, 118, 98],   near: [126, 236, 178] },
    coral: { far: [128, 44, 58],   near: [255, 118, 94]  }
  };

  function lerp(a, b, t) { return a + (b - a) * t; }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function mixColor(a, b, t) {
    return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
  }

  function Rosette(canvas, opts) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.opts = Object.assign({
      strands: 46,
      samples: 190,
      order: 0.7,
      palette: "gold",
      spin: 0.16,
      zoom: 1,
      shiftX: 0,
      scroll: false
    }, opts || {});

    this.cur = {
      order: this.opts.order,
      far: PALETTES[this.opts.palette].far.slice(),
      near: PALETTES[this.opts.palette].near.slice()
    };
    this.target = {
      order: this.opts.order,
      far: this.cur.far.slice(),
      near: this.cur.near.slice()
    };

    this.angle = 0.6;
    this.boost = 0;
    this.pointer = { x: 0, y: 0, sx: 0, sy: 0 };
    this.time = 0;
    this.running = false;
    this.visible = true;
    this.last = 0;

    var n = this.opts.strands * (this.opts.samples + 1);
    this.px = new Float32Array(n);
    this.py = new Float32Array(n);
    this.pz = new Float32Array(n);

    this._bindEvents();
    this._resize();
    this.start();
  }

  Rosette.prototype._bindEvents = function () {
    var self = this;

    window.addEventListener("pointermove", function (e) {
      self.pointer.x = (e.clientX / window.innerWidth) * 2 - 1;
      self.pointer.y = (e.clientY / window.innerHeight) * 2 - 1;
      if (!self.running) self._frame(performance.now());
    }, { passive: true });

    if ("ResizeObserver" in window) {
      this._ro = new ResizeObserver(function () { self._resize(); });
      this._ro.observe(this.canvas);
    } else {
      window.addEventListener("resize", function () { self._resize(); });
    }

    if ("IntersectionObserver" in window) {
      this._io = new IntersectionObserver(function (entries) {
        self.visible = entries[0].isIntersecting;
        if (self.visible) self.start();
      });
      this._io.observe(this.canvas);
    }
  };

  Rosette.prototype._resize = function () {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = this.canvas.clientWidth;
    var h = this.canvas.clientHeight;
    this.dpr = dpr;
    this.w = w;
    this.h = h;
    this.canvas.width = Math.max(1, Math.round(w * dpr));
    this.canvas.height = Math.max(1, Math.round(h * dpr));
    if (!this.running) this._frame(performance.now());
  };

  Rosette.prototype.set = function (next) {
    if (typeof next.order === "number") this.target.order = clamp(next.order, 0, 1);
    if (next.palette && PALETTES[next.palette]) {
      this.target.far = PALETTES[next.palette].far.slice();
      this.target.near = PALETTES[next.palette].near.slice();
    }
    this.start();
  };

  Rosette.prototype.burst = function () {
    this.boost = reduceMotion ? 0 : 7;
    this.start();
  };

  Rosette.prototype.start = function () {
    if (this.running || reduceMotion) {
      if (reduceMotion) this._frame(performance.now());
      return;
    }
    this.running = true;
    this.last = performance.now();
    var self = this;
    (function loop(now) {
      if (!self.running) return;
      if (!self.visible) { self.running = false; return; }
      self._frame(now);
      requestAnimationFrame(loop);
    })(this.last);
  };

  Rosette.prototype._frame = function (now) {
    var dt = clamp((now - this.last) / 1000, 0, 0.05);
    this.last = now;
    this.time += dt;

    // ease toward targets
    var k = 1 - Math.pow(0.02, dt);
    this.cur.order = lerp(this.cur.order, this.target.order, k * 0.9);
    for (var i = 0; i < 3; i++) {
      this.cur.far[i] = lerp(this.cur.far[i], this.target.far[i], k);
      this.cur.near[i] = lerp(this.cur.near[i], this.target.near[i], k);
    }
    this.pointer.sx = lerp(this.pointer.sx, this.pointer.x, k * 0.6);
    this.pointer.sy = lerp(this.pointer.sy, this.pointer.y, k * 0.6);
    this.boost *= Math.pow(0.18, dt);
    if (this.boost < 0.01) this.boost = 0;

    this.angle += dt * (this.opts.spin + this.boost);
    this._draw();
  };

  Rosette.prototype._draw = function () {
    var ctx = this.ctx;
    var W = this.canvas.width, H = this.canvas.height, dpr = this.dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);

    var order = this.cur.order;
    var chaos = 1 - order;
    var t = this.time;
    var S = Math.min(W, H) * 0.215 * this.opts.zoom;
    var cx = W / 2 + this.opts.shiftX * W;
    var cy = H / 2;
    var focal = 3.4;

    var scrollY = this.opts.scroll ? (window.scrollY || 0) : 0;
    var yaw = this.angle + this.pointer.sx * 0.55 + scrollY * 0.0016;
    var pitch = 0.95 + this.pointer.sy * 0.35 + scrollY * 0.0007;
    var cyw = Math.cos(yaw), syw = Math.sin(yaw);
    var cp = Math.cos(pitch), sp = Math.sin(pitch);

    var strands = this.opts.strands, samples = this.opts.samples;
    var amp1 = lerp(0.20, 0.07, order);       // main weave depth
    var amp2 = chaos * 0.20;                    // turbulence layer
    var amp3 = 0.05 + chaos * 0.06;             // slow breathing of the ring
    var pulse = t * (0.6 + chaos * 0.9);

    var px = this.px, py = this.py, pz = this.pz;
    var idx = 0;

    for (var j = 0; j < strands; j++) {
      var vj = (j / strands) * TAU;
      for (var s = 0; s <= samples; s++) {
        var u = (s / samples) * TAU;
        // braid: each strand also winds around the tube three times
        var v = vj + 3 * u;
        var r = 0.42 * (1
          + amp1 * Math.sin(5 * u + 2 * vj + pulse)
          + amp2 * Math.sin(11 * u - 3 * vj - pulse * 1.7));
        var R = 1 + amp3 * Math.sin(3 * u + t * 0.5);
        var ring = R + r * Math.cos(v);
        var x = ring * Math.cos(u);
        var y = ring * Math.sin(u);
        var z = r * Math.sin(v);

        // rotate: yaw around Z, then pitch around X
        var x1 = x * cyw - y * syw;
        var y1 = x * syw + y * cyw;
        var y2 = y1 * cp - z * sp;
        var z2 = y1 * sp + z * cp;

        var sc = focal / (focal + z2);
        px[idx] = cx + x1 * sc * S;
        py[idx] = cy + y2 * sc * S;
        pz[idx] = z2;
        idx++;
      }
    }

    // Bucket segments by depth so each bucket is one stroke call.
    var BINS = 6;
    ctx.globalCompositeOperation = "lighter";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    var zMin = -1.65, zMax = 1.65;
    // far buckets first, near buckets last
    for (var bin = BINS - 1; bin >= 0; bin--) {
      var depth = 1 - (bin + 0.5) / BINS;              // 1 = near, 0 = far
      var col = mixColor(this.cur.far, this.cur.near, Math.pow(depth, 0.85));
      ctx.strokeStyle = "rgba(" + (col[0] | 0) + "," + (col[1] | 0) + "," + (col[2] | 0) + "," +
        lerp(0.10, 0.85, Math.pow(depth, 1.4)).toFixed(3) + ")";
      ctx.lineWidth = lerp(0.6, 1.5, depth) * dpr;
      ctx.beginPath();

      for (var jj = 0; jj < strands; jj++) {
        var base = jj * (samples + 1);
        for (var ss = 0; ss < samples; ss++) {
          var a = base + ss, c = a + 1;
          var zz = (pz[a] + pz[c]) * 0.5;
          // near (small z) -> low bin index
          var q = clamp((zz - zMin) / (zMax - zMin), 0, 0.9999);
          if ((q * BINS | 0) !== bin) continue;
          ctx.moveTo(px[a], py[a]);
          ctx.lineTo(px[c], py[c]);
        }
      }
      ctx.stroke();
    }
    ctx.globalCompositeOperation = "source-over";
  };

  function init() {
    var canvases = document.querySelectorAll("canvas[data-rosette]");
    for (var i = 0; i < canvases.length; i++) {
      var el = canvases[i];
      el._rosette = new Rosette(el, {
        order: el.dataset.order !== undefined ? parseFloat(el.dataset.order) : 0.7,
        palette: el.dataset.palette || "gold",
        zoom: el.dataset.zoom ? parseFloat(el.dataset.zoom) : 1,
        shiftX: el.dataset.shiftx ? parseFloat(el.dataset.shiftx) : 0,
        spin: el.dataset.spin ? parseFloat(el.dataset.spin) : 0.16,
        scroll: el.dataset.scroll === "true"
      });
    }
  }

  window.Rosette = Rosette;
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
