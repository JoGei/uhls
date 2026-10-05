"""Display-only replacement for graphviz.Source in the browser cookbook.

This is NOT a replacement for graphviz's file-rendering or subprocess API.
The JavaScript module is fetched from a CDN.
"""
import anywidget
import traitlets


class DotGraph(anywidget.AnyWidget):
    dot = traitlets.Unicode().tag(sync=True)
    _esm = """
    // Major-version pin for this prototype. Pin an exact version for deployment.
    const vizPromise = import("https://esm.sh/@viz-js/viz@3.31.0")
      .then(({ instance }) => instance());

    export default {
      async render({ model, el }) {
        el.style.overflow = "auto";
        el.style.maxHeight = "650px";
        el.textContent = "Loading Graphviz…";
        let viz;
        try {
          viz = await vizPromise;
        } catch (error) {
          el.textContent = "Graphviz could not load: " + String(error);
          return;
        }
        const update = () => {
          try {
            el.replaceChildren(viz.renderSVGElement(model.get("dot")));
          } catch (error) {
            el.textContent = "Graphviz could not render: " + String(error);
          }
        };
        update();
        model.on("change:dot", update);
        return () => model.off("change:dot", update);
      }
    };
    """


def Source(dot_source: str) -> DotGraph:
    """Keep the cookbook's existing Source(dot_string) display calls."""
    if not isinstance(dot_source, str):
        raise TypeError("Source expects a DOT string")
    return DotGraph(dot=dot_source)
