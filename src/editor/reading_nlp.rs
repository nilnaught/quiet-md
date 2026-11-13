//! 阅读标注: the reading mode's NLP overlay (`reader_nlp`): POS colours,
//! connective underlines, non-finite dotted lines and core-predicate dots,
//! plus stepping through the document a sentence or paragraph at a time.
//!
//! Nothing here touches the document: the analysis is derived from
//! `Document::plain_text()` and painted from the block `RichText`s, so no
//! Mark is ever written. The plain text walks blocks exactly like
//! `Document::plain_text` (one `\n` between non-empty blocks, table cells
//! included) and yields a global byte offset for every block, so the token
//! ranges map back onto each block. Tokens that do not fall on block text
//! exactly (an equation's source replaces its `U+FFFC`) are dropped.

use super::keymap as k;
use super::text::{RichTextElement, TextLayout};
use super::view::Editor;
use crate::model::{Block, BlockId, BlockKind, Document};
use crate::theme;
use gpui::*;
use std::collections::HashMap;
use std::ops::Range;
use std::rc::Rc;

/// Default POS colours (NOUN / VERB / ADJ / ADV).
pub const NOUN: u32 = 0x2456AE;
pub const VERB: u32 = 0x8E1F2C;
pub const ADJ: u32 = 0x00783E;
pub const ADV: u32 = 0x7A4A1E;

/// How many annotation repaints one analysis waits for the models (30ms each).
const READY_TRIES: u16 = 400;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MarkOpts {
    pub nouns: bool,
    pub verbs: bool,
    pub adjs: bool,
    pub advs: bool,
    pub connectives: bool,
    pub nonfinite: bool,
    pub core: bool,
    pub dim: bool,
}

impl MarkOpts {
    pub fn from_prefs() -> Self {
        let p = crate::settings::prefs();
        MarkOpts { nouns: p.reader_nouns, verbs: p.reader_verbs, adjs: p.reader_adjs, advs: p.reader_advs, connectives: p.reader_connectives, nonfinite: p.reader_nonfinite, core: p.reader_core, dim: p.reader_dim }
    }
}

/// One token of a block, in that block's byte offsets.
#[derive(Clone, Debug)]
pub struct Mark {
    pub range: Range<usize>,
    pub upos: String,
    pub core: bool,
    pub connective: bool,
    pub nonfinite: bool,
}

/// What one block's text is painted with.
#[derive(Clone, Debug)]
pub struct Overlay {
    pub marks: Rc<Vec<Mark>>,
    /// 未读淡化: fade the text from this byte offset on (`None`: all read).
    pub dim_from: Option<usize>,
    /// The current 逐步阅读 unit's range in this block.
    pub unit: Option<Range<usize>>,
    pub opts: MarkOpts,
}

/// The colour a class paints with, `None` when its switch is off (or it has
/// no colour: PRON and OTHER stay the body colour).
pub fn mark_color(upos: &str, o: &MarkOpts) -> Option<Hsla> {
    match upos {
        "NOUN" if o.nouns => Some(theme::c(NOUN)),
        "VERB" if o.verbs => Some(theme::c(VERB)),
        "ADJ" if o.adjs => Some(theme::c(ADJ)),
        "ADV" if o.advs => Some(theme::c(ADV)),
        _ => None,
    }
}

/// The unit the current step selects, in plain-text offsets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitSpan {
    pub start: usize,
    pub end: usize,
    /// A paragraph unit (else a sentence).
    pub paragraph: bool,
}

/// Everything the overlay needs besides the document, built when the text,
/// the unit or the step changes.
#[derive(Clone, Default)]
pub struct Analysis {
    /// Block text -> global byte offset (`None`: not part of the plain text).
    pub offsets: HashMap<BlockId, Option<usize>>,
    pub tokens: Vec<reader_nlp::TokenSpan>,
    pub units: Vec<UnitSpan>,
    pub unit: Option<usize>,
    /// 未读淡化: the plain offset of the current unit's start.
    pub dim_at: Option<usize>,
    #[allow(dead_code)]
    pub neural: bool,
}

impl Analysis {
    /// The overlay for one block, if it has any.
    pub fn overlay(&self, id: BlockId, text: &crate::model::RichText, opts: MarkOpts) -> Option<Overlay> {
        let base = (*self.offsets.get(&id)?)?;
        let mut marks = vec![];
        for t in &self.tokens {
            if t.range.start < base || t.range.end > base + text.text.len() {
                continue;
            }
            let r = t.range.start - base..t.range.end - base;
            // Inline objects (an equation's U+FFFC) are not words: drop the
            // token rather than paint over the object.
            if r.end > text.text.len() || !has_text(text, &r) {
                continue;
            }
            if !t.is_core && !t.is_connective && !t.is_nonfinite && mark_color(&t.upos, &opts).is_none() {
                continue;
            }
            marks.push(Mark { range: r, upos: t.upos.clone(), core: t.is_core, connective: t.is_connective, nonfinite: t.is_nonfinite });
        }
        let unit = if self.unit.is_some() {
            self.unit.and_then(|u| self.units.get(u)).and_then(|u| {
                let a = u.start.max(base);
                let b = u.end.min(base + text.text.len());
                (a < b).then(|| a - base..b - base)
            })
        } else {
            None
        };
        let dim_from = if opts.dim { self.dim_at.map(|d| d.saturating_sub(base)).filter(|d| *d < text.text.len()) } else { None };
        if marks.is_empty() && unit.is_none() && dim_from.is_none() {
            return None;
        }
        Some(Overlay { marks: Rc::new(marks), dim_from, unit, opts })
    }
}

/// No byte of `r` belongs to an inline object (an equation's `U+FFFC`).
fn has_text(text: &crate::model::RichText, r: &Range<usize>) -> bool {
    for (span, m) in text.spans() {
        if span.start < r.end && span.end > r.start && m.inline.is_some() {
            return false;
        }
    }
    true
}

/// The plain text plus every block's offset into it, walking blocks the way
/// `Document::plain_text` does. `None` for a block whose text is empty or
/// which `plain_text` skips.
fn plain_with_offsets(doc: &Document) -> (String, HashMap<BlockId, Option<usize>>) {
    let mut out = String::new();
    let mut offsets = HashMap::new();
    fn walk(blocks: &[Block], out: &mut String, offsets: &mut HashMap<BlockId, Option<usize>>) {
        for b in blocks {
            if !b.text.is_empty() {
                if !out.is_empty() {
                    out.push('\n');
                }
                offsets.insert(b.id, Some(out.len()));
                out.push_str(&b.text.plain_text());
            } else {
                offsets.entry(b.id).or_insert(None);
            }
            walk(&b.children, out, offsets);
            if let BlockKind::Table(t) = &b.kind {
                for row in &t.rows {
                    for cell in row {
                        walk(&cell.blocks, out, offsets);
                    }
                }
            }
        }
    }
    walk(&doc.blocks, &mut out, &mut offsets);
    (out, offsets)
}

impl Editor {
    /// 阅读标注 is on: 阅读模式 and the 启用阅读标注 switch.
    pub fn reader_marks_on(&self) -> bool {
        self.read_only && self.reader_active
    }

    /// (Re)analyse the document in the background; the models may still be
    /// loading, in which case the fallback's marks show now and the neural
    /// ones replace them (`poll_reader_nlp`).
    pub fn reset_reader_nlp(&mut self) {
        self.reader_nlp = None;
        self.reader_key = None;
        self.reader_focus = None;
        self.reader_gen = self.reader_gen.wrapping_add(1);
    }

    /// After a render: analyse if the text or the step changed, and repaint
    /// when the models finish loading.
    pub(super) fn after_frame(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        if !self.reader_marks_on() {
            return;
        }
        let (text, offsets) = plain_with_offsets(&self.core.doc);
        let paragraph = self.reader_units_paragraphs;
        let step = self.reader_step;
        let key = (hash_str(&text), paragraph, step, self.core.generation);
        if self.reader_key == Some(key) {
            return;
        }
        self.reader_key = Some(key);
        let engine = crate::settings::nlp();
        let ready = engine.neural_ready();
        let my_gen = self.reader_gen;
        cx.spawn(async move |this, cx| {
            let computed = cx.background_executor().spawn(async move { compute(&engine, &text) }).await;
            let (tokens, units) = computed;
            let ok = this.update(cx, |ed, cx| {
                // A newer document (or a model switch) superseded this one.
                if ed.reader_gen != my_gen || ed.reader_key != Some(key) {
                    return false;
                }
                ed.apply_reader_nlp(tokens, units, offsets, paragraph, step, ready, cx);
                true
            });
            if !ok.is_ok_and(|applied| applied) {
                return;
            }
            // The models load in the background: analyse once more when they
            // are in, so the fallback's marks give way to the neural ones.
            if !ready {
                for _ in 0..READY_TRIES {
                    cx.background_executor().timer(std::time::Duration::from_millis(30)).await;
                    if crate::settings::nlp().neural_ready() {
                        break;
                    }
                }
                // Analyse again: with the models if they made it, else the
                // next render's key change retries.
                let _ = this.update(cx, |ed, cx| {
                    if ed.reader_gen == my_gen && ed.reader_key == Some(key) {
                        ed.reader_key = None;
                        cx.notify();
                    }
                });
            }
        })
        .detach();
    }

    /// Apply a finished analysis: place the first unit (or the step the reader
    /// last chose) and repaint.
    #[allow(clippy::too_many_arguments)]
    fn apply_reader_nlp(&mut self, tokens: Vec<reader_nlp::TokenSpan>, units: Vec<UnitSpan>, offsets: HashMap<BlockId, Option<usize>>, paragraph: bool, step: usize, ready: bool, cx: &mut Context<Self>) {
        let units: Vec<UnitSpan> = units.into_iter().filter(|u| u.paragraph == paragraph).collect();
        let n = units.len();
        let at = if self.reader_step_pending { self.reader_anchor } else { None };
        self.reader_step_pending = false;
        let step = match at {
            Some(t) => units.iter().position(|u| u.start <= t && t < u.end).unwrap_or(step.min(n.saturating_sub(1))),
            None => step.min(n.saturating_sub(1)),
        };
        self.reader_step = step;
        let current = units.get(step).copied();
        let a = Analysis { offsets, tokens, units, unit: current.map(|_| step), dim_at: current.map(|u| u.start), neural: ready };
        self.reader_nlp = Some(Rc::new(a));
        self.reader_units_paragraphs = paragraph;
        self.cache.borrow_mut().clear();
        // A fresh analysis (or the first unit of a document): bring it into view.
        if current.is_some() && self.reader_focus.is_none() {
            self.reader_focus = Some(());
            self.reveal_reader_step();
        }
        cx.notify();
    }

    /// Move the page to the current unit's block and centre it there.
    fn reveal_reader_step(&mut self) {
        let Some(a) = self.reader_nlp.clone() else { return };
        let Some(u) = a.unit.and_then(|i| a.units.get(i)).copied() else { return };
        if let Some((id, at)) = unit_pos(&a.offsets, &self.core.doc, u.start) {
            self.reader_show = Some((id, at));
            self.reveal = Some((super::render::Reveal::Find(crate::editor::core::Pos::new(id, at)), 0));
        }
    }

    /// Current unit's index and count, and whether the models answered.
    #[allow(dead_code)]
    pub fn reader_progress(&self) -> Option<(usize, usize, bool)> {
        let a = self.reader_nlp.as_ref()?;
        (!a.units.is_empty()).then(|| (a.unit.unwrap_or(0), a.units.len(), a.neural))
    }

    pub(super) fn on_reader_next(&mut self, _: &k::ReaderNext, _: &mut Window, cx: &mut Context<Self>) {
        if !self.read_only {
            return;
        }
        if !self.reader_marks_on() {
            return self.reader_need_mode(cx);
        }
        self.reader_step_by(1, cx);
    }

    pub(super) fn on_reader_prev(&mut self, _: &k::ReaderPrev, _: &mut Window, cx: &mut Context<Self>) {
        if !self.read_only {
            return;
        }
        if !self.reader_marks_on() {
            return self.reader_need_mode(cx);
        }
        self.reader_step_by(-1, cx);
    }

    pub(super) fn on_reader_toggle_unit(&mut self, _: &k::ReaderToggleUnit, _: &mut Window, cx: &mut Context<Self>) {
        if !self.read_only {
            return;
        }
        if !self.reader_marks_on() {
            return self.reader_need_mode(cx);
        }
        // 句 ⇄ 段: the step stays at the unit under the one now shown.
        let at = self.reader_nlp.as_ref().and_then(|a| a.unit.and_then(|u| a.units.get(u))).map(|u| u.start);
        self.reader_units_paragraphs = !self.reader_units_paragraphs;
        self.reader_anchor = at;
        self.reader_step_pending = at.is_some();
        self.reader_key = None;
        cx.notify();
    }

    /// Space in 阅读模式 without the overlay: the old reminder.
    fn reader_need_mode(&mut self, cx: &mut Context<Self>) {
        use crate::editor_ui::ToastKind;
        self.toast_kind(ToastKind::Info, super::reading::READ_ONLY_HINT, cx);
    }

    /// Step `delta` units, stop at either end, and keep the unit on screen.
    fn reader_step_by(&mut self, delta: i32, cx: &mut Context<Self>) {
        if self.core.block_sel.is_some() {
            self.core.clear_block_sel();
        }
        let Some(a) = self.reader_nlp.clone() else {
            self.reader_key = None;
            return;
        };
        let n = a.units.len();
        let step = (a.unit.unwrap_or(0) as i32 + delta).clamp(0, n as i32 - 1) as usize;
        if step == a.unit.unwrap_or(0) {
            return;
        }
        self.reader_step = step;
        // Rebuild the overlays with the new current unit (the marks are the same).
        let mut next = (*a).clone();
        next.unit = Some(step);
        next.dim_at = next.units.get(step).map(|u| u.start);
        self.reader_nlp = Some(Rc::new(next));
        self.cache.borrow_mut().clear();
        self.reader_focus = Some(());
        self.reveal_reader_step();
        cx.notify();
    }

    /// Step back to the first unit (a document switch starts a fresh read).
    pub(super) fn reader_start(&mut self) {
        self.reader_step = 0;
        self.reader_focus = None;
    }
}

fn compute(engine: &reader_nlp::NlpEngine, text: &str) -> (Vec<reader_nlp::TokenSpan>, Vec<UnitSpan>) {
    let a = engine.analyze(text);
    let mut tokens = Vec::new();
    for s in &a.sentences {
        tokens.extend(s.tokens.iter().cloned());
    }
    let mut units: Vec<UnitSpan> = a.sentences.iter().map(|s| UnitSpan { start: s.range.start, end: s.range.end, paragraph: false }).collect();
    units.extend(a.paragraphs.iter().map(|p| UnitSpan { start: p.range.start, end: p.range.end, paragraph: true }));
    (tokens, units)
}

fn hash_str(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    s.hash(&mut h);
    h.finish()
}

/// The block and offset that holds plain offset `at`.
fn unit_pos(offsets: &HashMap<BlockId, Option<usize>>, doc: &Document, at: usize) -> Option<(BlockId, usize)> {
    let mut best: Option<(usize, BlockId)> = None;
    doc.for_each_block(|b| {
        let Some(Some(o)) = offsets.get(&b.id) else { return };
        if *o <= at && !b.text.is_empty() {
            let end = *o + b.text.plain_text().len();
            let dist = at.saturating_sub(end);
            if best.is_none_or(|(d, _)| dist < d) {
                best = Some((dist, b.id));
            }
        }
    });
    let (_, id) = best?;
    let base = (*offsets.get(&id)?)?;
    let len = doc_text_len(doc, id);
    Some((id, at.saturating_sub(base).min(len)))
}

fn doc_text_len(doc: &Document, id: BlockId) -> usize {
    let mut n = 0;
    doc.for_each_block(|b| {
        if b.id == id {
            n = b.text.plain_text().len();
        }
    });
    n
}

/// Colour TextRuns with POS classes. Splits existing runs at mark boundaries;
/// connective / non-finite / core marks are still painted in `paint_reading`.
pub(super) fn color_runs(text_len: usize, runs: Vec<TextRun>, overlay: &Overlay) -> Vec<TextRun> {
    if overlay.marks.is_empty() || text_len == 0 || runs.is_empty() {
        return runs;
    }
    // Flatten to (start, end, run) then split on mark edges.
    let mut pieces: Vec<(usize, usize, TextRun)> = Vec::new();
    let mut off = 0usize;
    for run in runs {
        let end = (off + run.len).min(text_len);
        if off < end {
            pieces.push((off, end, run));
        }
        off = end;
    }
    let mut cuts: Vec<usize> = pieces.iter().flat_map(|(a, b, _)| [*a, *b]).collect();
    for m in overlay.marks.iter() {
        cuts.push(m.range.start.min(text_len));
        cuts.push(m.range.end.min(text_len));
    }
    cuts.sort();
    cuts.dedup();
    let mut out = Vec::new();
    for w in cuts.windows(2) {
        let (a, b) = (w[0], w[1]);
        if a >= b {
            continue;
        }
        let Some((_, _, src)) = pieces.iter().find(|(s, e, _)| *s <= a && b <= *e) else {
            continue;
        };
        let mut run = src.clone();
        run.len = b - a;
        if let Some(m) = overlay.marks.iter().find(|m| m.range.start <= a && b <= m.range.end)
            && let Some(c) = mark_color(&m.upos, &overlay.opts)
        {
            run.color = c;
        }
        out.push(run);
    }
    out
}

// ---- painting

impl RichTextElement {
    /// The overlay's tint beneath a unit, the faded tail of 未读淡化 and the
    /// underlines / dotted lines / dots of the marks. Called with the glyphs
    /// painted; `origin` is where line painting started.
    pub(super) fn paint_reading(&self, layout: &TextLayout, origin: Point<Pixels>, window: &mut Window) {
        let Some(o) = &self.overlay else { return };
        let lh = layout.line_height;
        // The current sentence / paragraph: a pale band behind its rows.
        if let Some(u) = &o.unit {
            let bg = theme::rgbaf(0x24, 0x56, 0xAE, 0.06);
            for r in layout.rects_for(u.clone()) {
                let h = (px(self.spec.size * 1.5)).min(lh);
                let top = r.top() + (lh - h) / 2.;
                window.paint_quad(fill(Bounds::new(point(r.left() - px(2.), top), size(r.size.width + px(4.), h)), bg).corner_radii(px(3.)));
            }
        }
        let text = &self.text.text;
        let body = self.spec.color;
        for m in o.marks.iter() {
            let ink = mark_color(&m.upos, &o.opts).unwrap_or(body);
            let faded = o.dim_from.is_some_and(|d| m.range.start >= d);
            let a = if faded { 0.45 } else { 1. };
            let rects = layout.rects_for(m.range.clone());
            for r in &rects {
                if o.opts.connectives && m.connective {
                    let y = r.bottom() - px(1.5);
                    window.paint_quad(fill(Bounds::new(point(r.left(), y), size(r.size.width, px(1.2))), ink.opacity(a)));
                }
                if o.opts.nonfinite && m.nonfinite {
                    // GPUI has no dashed underline: a 2px dotted line of 1.4px
                    // segments along the glyphs' bottom.
                    let y = r.bottom() - px(1.);
                    let mut x = r.left();
                    while x < r.right() {
                        window.paint_quad(fill(Bounds::new(point(x, y), size(px(1.4).min(r.right() - x), px(2.))), ink.opacity(0.7 * a)));
                        x += px(3.4);
                    }
                }
                if o.opts.core && m.core {
                    paint_core_dot(window, r, text, &m.range, ink.opacity(a));
                }
            }
        }
        // 未读淡化: the unread tail of this block at 45%. Glyph colours are
        // settled when the line is shaped, so the fade is a paper-coloured
        // wash over the rows: 55% of it leaves the text at .45.
        if let Some(d) = o.dim_from.filter(|d| *d <= text.len()) {
            let paper = self.spec.paper.unwrap_or_else(crate::editor_theme::Theme::bg_body);
            let a = 0.55 * (if theme::dark() { 0.8 } else { 1. });
            for r in layout.rects_for(d..text.len()) {
                window.paint_quad(fill(Bounds::new(point(r.left(), r.top()), size(r.size.width, r.size.height)), paper.opacity(a)));
            }
        }
        let _ = origin;
    }
}

/// A core predicate: one dot under every Han character, else one under the
/// word's centre.
fn paint_core_dot(window: &mut Window, row: &Bounds<Pixels>, text: &str, range: &Range<usize>, color: Hsla) {
    let y = row.bottom() + px(0.5);
    let size_of = px(2.6);
    let mut han = 0;
    for (i, ch) in text[range.start.min(text.len())..range.end.min(text.len())].char_indices() {
        if !is_han(ch) {
            continue;
        }
        // Every Han glyph is full-width; place the dot under its centre the
        // way the browser would (one emphasis dot per character).
        let frac = (i as f32 + ch.len_utf8() as f32 / 2.) / (range.end - range.start).max(1) as f32;
        let x = row.left() + row.size.width * frac.min(1.);
        window.paint_quad(fill(Bounds::new(point(x - size_of / 2., y), size(size_of, size_of)), color).corner_radii(size_of / 2.));
        han += 1;
    }
    if han == 0 {
        let x = row.center().x;
        window.paint_quad(fill(Bounds::new(point(x - px(1.6), y), size(px(3.2), px(3.2))), color).corner_radii(px(1.6)));
    }
}

fn is_han(ch: char) -> bool {
    ('\u{2E80}'..='\u{9FFF}').contains(&ch) || ('\u{F900}'..='\u{FFEF}').contains(&ch)
}
