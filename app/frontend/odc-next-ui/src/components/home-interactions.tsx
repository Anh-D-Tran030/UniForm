"use client";

import { useEffect, useRef } from "react";

/**
 * Behavior-only client leaf for the marketing home page. It attaches DOM
 * effects after hydration so the page itself can stay a Server Component:
 *  - scroll-triggered reveals for [data-reveal]
 *  - count-up for [data-count]
 *  - pointer spotlight for [data-spotlight]
 *  - hero pointer glow [data-hero-glow] + parallax tilt [data-tilt]
 *  - a top scroll-progress meter
 * Everything is gated on prefers-reduced-motion and degrades to fully visible
 * content when JavaScript is unavailable (hidden states require html.js-reveal).
 */
export function HomeInteractions() {
  const progressRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const root = document.documentElement;

    const animateCount = (el: HTMLElement) => {
      const target = Number(el.dataset.count ?? "0");
      const decimals = Number(el.dataset.decimals ?? "0");
      const prefix = el.dataset.prefix ?? "";
      const suffix = el.dataset.suffix ?? "";
      const render = (value: number) => {
        el.textContent = `${prefix}${value.toFixed(decimals)}${suffix}`;
      };
      if (prefersReduced) {
        render(target);
        return;
      }
      const duration = 1400;
      const start = performance.now();
      const tick = (now: number) => {
        const progress = Math.min(1, (now - start) / duration);
        const eased = 1 - Math.pow(1 - progress, 3);
        render(target * eased);
        if (progress < 1) {
          requestAnimationFrame(tick);
        } else {
          render(target);
        }
      };
      render(0);
      requestAnimationFrame(tick);
    };

    // Reduced motion: leave content visible, resolve any counters, do nothing else.
    if (prefersReduced) {
      document.querySelectorAll<HTMLElement>("[data-count]").forEach(animateCount);
      return;
    }

    root.classList.add("js-reveal");

    const counted = new WeakSet<HTMLElement>();
    const runCounts = (scope: HTMLElement) => {
      if (scope.hasAttribute("data-count") && !counted.has(scope)) {
        counted.add(scope);
        animateCount(scope);
      }
      scope.querySelectorAll<HTMLElement>("[data-count]").forEach((c) => {
        if (!counted.has(c)) {
          counted.add(c);
          animateCount(c);
        }
      });
    };

    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) {
            continue;
          }
          const el = entry.target as HTMLElement;
          el.classList.add("is-visible");
          runCounts(el);
          io.unobserve(el);
        }
      },
      { threshold: 0.18, rootMargin: "0px 0px -8% 0px" },
    );

    document.querySelectorAll<HTMLElement>("[data-reveal]").forEach((el) => io.observe(el));
    document.querySelectorAll<HTMLElement>("[data-count]").forEach((el) => {
      if (!el.closest("[data-reveal]")) {
        io.observe(el);
      }
    });

    const onMove = (event: PointerEvent) => {
      const target = event.target as HTMLElement | null;
      const card = target?.closest?.("[data-spotlight]") as HTMLElement | null;
      if (!card) {
        return;
      }
      const rect = card.getBoundingClientRect();
      card.style.setProperty("--mx", `${((event.clientX - rect.left) / rect.width) * 100}%`);
      card.style.setProperty("--my", `${((event.clientY - rect.top) / rect.height) * 100}%`);
    };
    document.addEventListener("pointermove", onMove, { passive: true });

    const hero = document.querySelector<HTMLElement>("[data-hero]");
    const glow = document.querySelector<HTMLElement>("[data-hero-glow]");
    const tilt = document.querySelector<HTMLElement>("[data-tilt]");

    const onHeroMove = (event: PointerEvent) => {
      if (!hero) {
        return;
      }
      const rect = hero.getBoundingClientRect();
      const px = (event.clientX - rect.left) / rect.width;
      const py = (event.clientY - rect.top) / rect.height;
      if (glow) {
        glow.classList.add("is-live");
        glow.style.setProperty("--mx", `${px * 100}%`);
        glow.style.setProperty("--my", `${py * 100}%`);
      }
      if (tilt) {
        tilt.style.setProperty("--rx", `${(px - 0.5) * 9}deg`);
        tilt.style.setProperty("--ry", `${(0.5 - py) * 9}deg`);
      }
    };
    const onHeroLeave = () => {
      glow?.classList.remove("is-live");
      if (tilt) {
        tilt.style.setProperty("--rx", "0deg");
        tilt.style.setProperty("--ry", "0deg");
      }
    };
    hero?.addEventListener("pointermove", onHeroMove, { passive: true });
    hero?.addEventListener("pointerleave", onHeroLeave);

    let raf = 0;
    const onScroll = () => {
      if (raf) {
        return;
      }
      raf = requestAnimationFrame(() => {
        raf = 0;
        const bar = progressRef.current;
        if (!bar) {
          return;
        }
        const max = document.documentElement.scrollHeight - window.innerHeight;
        const ratio = max > 0 ? window.scrollY / max : 0;
        bar.style.setProperty("--scroll", String(Math.min(1, Math.max(0, ratio))));
      });
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    return () => {
      io.disconnect();
      document.removeEventListener("pointermove", onMove);
      hero?.removeEventListener("pointermove", onHeroMove);
      hero?.removeEventListener("pointerleave", onHeroLeave);
      window.removeEventListener("scroll", onScroll);
      if (raf) {
        cancelAnimationFrame(raf);
      }
      root.classList.remove("js-reveal");
    };
  }, []);

  return <div ref={progressRef} className="scroll-progress" aria-hidden="true" />;
}
