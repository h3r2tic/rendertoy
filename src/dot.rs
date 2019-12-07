//! Simple graphviz dot file format output.
//! Based on petgraph::dot.

use std::fmt::{self, Display, Write};

use petgraph::visit::GraphRef;

/// `Dot` implements output to graphviz .dot format for a graph.
///
/// Formatting and options are rather simple, this is mostly intended
/// for debugging. Exact output may change.
///
/// # Examples
///
/// ```
/// use petgraph::Graph;
/// use petgraph::dot::{Dot, Config};
///
/// let mut graph = Graph::<_, ()>::new();
/// graph.add_node("A");
/// graph.add_node("B");
/// graph.add_node("C");
/// graph.add_node("D");
/// graph.extend_with_edges(&[
///     (0, 1), (0, 2), (0, 3),
///     (1, 2), (1, 3),
///     (2, 3),
/// ]);
///
/// println!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel]));
pub struct Dot<G> {
    graph: G,
    dot_graph_attribs: Option<&'static str>,
}

static TYPE: [&'static str; 2] = ["graph", "digraph"];
static EDGE: [&'static str; 2] = ["--", "->"];
static INDENT: &'static str = "    ";
static GRAPH_PRELUDE: &'static str = r##"
bgcolor = "#505050";
node [
    shape=plaintext;

    fontname = "Segoe UI";
    margin = 0.15;
    fontcolor = "#f0f0f0";

    style = "filled";
    fillcolor = "#303030";
];
edge [
    color="#a0a0a0";
    penwidth = 0.5;
    arrowsize = 0.75;
    dir = back;
];"##;

impl<G> Dot<G>
where
    G: GraphRef,
{
    /// Create a `Dot` formatting wrapper with default configuration.
    pub fn new(graph: G, dot_graph_attribs: Option<&'static str>) -> Self {
        Dot {
            graph,
            dot_graph_attribs,
        }
    }
}

use petgraph::visit::{Data, GraphProp, NodeRef};
use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNodeReferences, NodeIndexable};

impl<G> Dot<G> {
    fn graph_fmt<NF, EF, NW, EW>(
        &self,
        g: G,
        f: &mut fmt::Formatter,
        mut node_fmt: NF,
        mut edge_fmt: EF,
    ) -> fmt::Result
    where
        G: NodeIndexable + IntoNodeReferences + IntoEdgeReferences,
        G: GraphProp,
        G: Data<NodeWeight = NW, EdgeWeight = EW>,
        NF: FnMut(&NW, &mut dyn FnMut(&dyn Display) -> fmt::Result) -> fmt::Result,
        EF: FnMut(&EW, &mut dyn FnMut(&dyn Display) -> fmt::Result) -> fmt::Result,
    {
        writeln!(f, "{} {{", TYPE[g.is_directed() as usize])?;
        f.write_str(GRAPH_PRELUDE)?;
        f.write_str("\n")?;
        if let Some(attribs) = self.dot_graph_attribs {
            f.write_str(attribs)?;
        }
        f.write_str("\n")?;

        // output all labels
        for (node_ordinal, node) in g.node_references().enumerate() {
            write!(f, "{}{}", INDENT, g.to_index(node.id()))?;
            write!(f, " [label=\"")?;
            node_fmt(node.weight(), &mut |d| Escaped(d).fmt(f))?;
            write!(f, "\"")?;

            // Highlight the root node
            if 0 == node_ordinal {
                write!(f, ", fillcolor = \"#603080\"")?;
            }

            writeln!(f, "]")?;
        }
        // output all edges
        for edge in g.edge_references() {
            write!(
                f,
                "{}{} {} {}",
                INDENT,
                g.to_index(edge.source()),
                EDGE[g.is_directed() as usize],
                g.to_index(edge.target())
            )?;
            write!(f, " [label=\"")?;
            edge_fmt(edge.weight(), &mut |d| Escaped(d).fmt(f))?;
            writeln!(f, "\"]")?;
        }

        writeln!(f, "}}")?;
        Ok(())
    }
}

impl<G> fmt::Display for Dot<G>
where
    G: IntoEdgeReferences + IntoNodeReferences + NodeIndexable + GraphProp,
    G::EdgeWeight: fmt::Display,
    G::NodeWeight: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.graph_fmt(self.graph, f, |n, cb| cb(n), |e, cb| cb(e))
    }
}

impl<G> fmt::Debug for Dot<G>
where
    G: IntoEdgeReferences + IntoNodeReferences + NodeIndexable + GraphProp,
    G::EdgeWeight: fmt::Debug,
    G::NodeWeight: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.graph_fmt(
            self.graph,
            f,
            |n, cb| cb(&DebugFmt(n)),
            |e, cb| cb(&DebugFmt(e)),
        )
    }
}

/// Escape for Graphviz
struct Escaper<W>(W);

impl<W> fmt::Write for Escaper<W>
where
    W: fmt::Write,
{
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for c in s.chars() {
            self.write_char(c)?;
        }
        Ok(())
    }

    fn write_char(&mut self, c: char) -> fmt::Result {
        match c {
            '"' | '\\' => self.0.write_char('\\')?,
            // \l is for left justified linebreak
            '\n' => return self.0.write_str("\\l"),
            _ => {}
        }
        self.0.write_char(c)
    }
}

/// Pass Display formatting through a simple escaping filter
struct Escaped<T>(T);

impl<T> fmt::Display for Escaped<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if f.alternate() {
            writeln!(&mut Escaper(f), "{:#}", &self.0)
        } else {
            write!(&mut Escaper(f), "{}", &self.0)
        }
    }
}

/// Pass Debug formatting to Display
struct DebugFmt<T>(T);

impl<T> fmt::Display for DebugFmt<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[test]
fn test_escape() {
    let mut buff = String::new();
    {
        let mut e = Escaper(&mut buff);
        let _ = e.write_str("\" \\ \n");
    }
    assert_eq!(buff, "\\\" \\\\ \\l");
}
