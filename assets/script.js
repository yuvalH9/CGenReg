const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        revealObserver.unobserve(entry.target);
      }
    });
  },
  {
    threshold: 0.15,
    rootMargin: "0px 0px -40px 0px",
  },
);

document.querySelectorAll("[data-reveal]").forEach((element) => {
  revealObserver.observe(element);
});

const trackEvent = (eventName, params = {}) => {
  if (typeof window.gtag === "function") {
    window.gtag("event", eventName, params);
  }
};

document.querySelectorAll("[data-analytics-event]").forEach((element) => {
  element.addEventListener("click", () => {
    const eventName = element.getAttribute("data-analytics-event");

    if (!eventName) {
      return;
    }

    trackEvent(eventName, {
      link_text: element.textContent?.trim() ?? "",
      link_url: element.getAttribute("href") ?? "",
    });
  });
});

const heroVideo = document.getElementById("hero-video");

if (heroVideo) {
  window.setTimeout(() => {
    const playPromise = heroVideo.play();

    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch(() => {
        // Some browsers may still block autoplay until the first interaction.
      });
    }
  }, 2500);
}

const copyButton = document.getElementById("copy-bibtex");

if (copyButton) {
  copyButton.addEventListener("click", async () => {
    const bibtex = document.querySelector("#bibtex code")?.textContent ?? "";

    try {
      await navigator.clipboard.writeText(bibtex);
      trackEvent("copy_bibtex_click", {
        location: "citation_section",
      });
      copyButton.textContent = "Copied";
      window.setTimeout(() => {
        copyButton.textContent = "Copy";
      }, 1800);
    } catch (error) {
      copyButton.textContent = "Copy failed";
      window.setTimeout(() => {
        copyButton.textContent = "Copy";
      }, 1800);
    }
  });
}

const qualitativeCarousel = document.querySelector(".qualitative-carousel");

if (qualitativeCarousel) {
  const slides = Array.from(
    qualitativeCarousel.querySelectorAll("[data-carousel-slide]"),
  );
  const dots = Array.from(
    qualitativeCarousel.querySelectorAll("[data-carousel-dot]"),
  );
  const prevButton = qualitativeCarousel.querySelector("[data-carousel-prev]");
  const nextButton = qualitativeCarousel.querySelector("[data-carousel-next]");
  let activeIndex = slides.findIndex((slide) => slide.classList.contains("is-active"));

  if (activeIndex < 0) {
    activeIndex = 0;
  }

  const renderSlide = (index) => {
    activeIndex = (index + slides.length) % slides.length;

    slides.forEach((slide, slideIndex) => {
      const isActive = slideIndex === activeIndex;
      slide.classList.toggle("is-active", isActive);
      slide.hidden = !isActive;
    });

    dots.forEach((dot, dotIndex) => {
      const isActive = dotIndex === activeIndex;
      dot.classList.toggle("is-active", isActive);
      dot.setAttribute("aria-selected", String(isActive));
    });
  };

  prevButton?.addEventListener("click", () => {
    renderSlide(activeIndex - 1);
  });

  nextButton?.addEventListener("click", () => {
    renderSlide(activeIndex + 1);
  });

  dots.forEach((dot, index) => {
    dot.addEventListener("click", () => {
      renderSlide(index);
    });
  });

  renderSlide(activeIndex);
}
