// orukeet-ane: Orukeet speech-to-text on the Apple Neural Engine via Core ML.
//
// Uses only Apple frameworks and Oruk AI's official Core ML bundle (6-bit palettised weights).
//   orukeet-ane <bundleDir> <cacheDir>            serve: one WAV path per stdin line -> one JSON line
//   orukeet-ane <bundleDir> <cacheDir> --warmup   compile/load once (first Neural Engine load is slow), then exit
//
// The model takes fixed 15 s windows. Longer audio is split into windows overlapping by 3 s and
// stitched at the word boundary nearest the middle of each overlap, using token timestamps.
import CoreML
import Foundation

let sampleRate = 16_000
let windowSamples = 15 * sampleRate
let strideSamples = 12 * sampleRate
let secondsPerFrame = 0.08                 // 10 ms mel hop x 8 subsampling
let encoderWidth = 1024, decoderWidth = 640, decoderLayers = 2
let maxSymbolsPerFrame = 10

struct Token { let id: Int; let time: Double }   // time = seconds from the start of the whole clip

/// Vocabulary boosting: while decoding, word-pieces that spell out a listed term get a bonus, so a close call
/// ("or a keet" vs "Orukeet") tips toward the term. Matching is by characters, so any split of the term into pieces
/// works. Only pieces the model already finds plausible (log-probability >= floor, within the top k) are boosted,
/// and a boosted piece must stay within `blankMargin` of "nothing said here", so boosting never invents speech.
struct Boost {
    var terms: [String] = []          // as the user spells them
    var beta: Float = 4.0             // log-probability bonus per matching piece (tuned on synthetic speech)
    var floor: Float = -6.0           // never boost pieces less likely than this
    var topK = 20
    var blankMargin: Float = 4.0      // a boosted piece may beat blank only if it was within this of blank already
    var keys: [[Character]] = []      // "▁" + lowercased term, spaces as "▁"

    init() {}
    init(terms: [String], options: [String: Any]) {
        self.terms = terms.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        if let v = options["beta"] as? Double { beta = Float(v) }
        if let v = options["floor"] as? Double { floor = Float(v) }
        if let v = options["top_k"] as? Int { topK = v }
        if let v = options["blank_margin"] as? Double { blankMargin = Float(v) }
        keys = self.terms.map { Array("▁" + $0.lowercased().replacingOccurrences(of: " ", with: "▁")) }
    }
    var isOn: Bool { !keys.isEmpty && beta > 0 }
}

/// A term being spelled out: which term, and how many characters of it have been emitted so far.
struct Partial: Hashable { let term: Int; let pos: Int }

func emit(_ object: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: object)
    FileHandle.standardOutput.write(data + Data("\n".utf8))
}

func array(_ shape: [Int], _ type: MLMultiArrayDataType) throws -> MLMultiArray {
    try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: type)
}

func zeros(_ shape: [Int]) throws -> MLMultiArray {
    let a = try array(shape, .float32)
    a.withUnsafeMutableBufferPointer(ofType: Float.self) { b, _ in b.initialize(repeating: 0) }
    return a
}

/// 16-bit PCM mono 16 kHz WAV (what FreeFlow records; the Python side converts anything else).
func readWav(_ url: URL) throws -> [Float] {
    let d = try Data(contentsOf: url)
    guard d.count >= 12, String(decoding: d[0..<4], as: UTF8.self) == "RIFF" else { throw NSError(domain: "not a WAV file", code: 1) }
    var i = 12
    while i + 8 <= d.count {
        let id = String(decoding: d[i..<i+4], as: UTF8.self)
        let size = Int(d[i+4..<i+8].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) })
        if id == "data" {
            return d[(i+8)..<min(i+8+size, d.count)].withUnsafeBytes { raw in
                raw.bindMemory(to: Int16.self).map { Float($0) / 32768 }
            }
        }
        i += 8 + size + (size & 1)
    }
    throw NSError(domain: "WAV has no data chunk", code: 2)
}

final class Orukeet {
    let pre, enc, dec, joint: MLModel
    let vocab: [String]
    var blank: Int32 { Int32(vocab.count) }

    init(bundle: URL, cache: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: cache, withIntermediateDirectories: true)
        func model(_ name: String) throws -> MLModel {
            let compiled = cache.appendingPathComponent("\(name).mlmodelc")
            if !fm.fileExists(atPath: compiled.path) {
                let tmp = try MLModel.compileModel(at: bundle.appendingPathComponent("\(name).mlpackage"))
                try fm.moveItem(at: tmp, to: compiled)
            }
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuAndNeuralEngine
            return try MLModel(contentsOf: compiled, configuration: cfg)
        }
        pre = try model("Preprocessor")
        enc = try model("Encoder")
        dec = try model("Decoder")
        // The boosting variant also returns the scores the stock model computes before its final argmax.
        let logits = fm.fileExists(atPath: bundle.appendingPathComponent("JointDecisionv3Logits.mlpackage").path)
        joint = try model(logits ? "JointDecisionv3Logits" : "JointDecisionv3")
        canBoost = logits
        let v = try JSONSerialization.jsonObject(with: Data(contentsOf: bundle.appendingPathComponent("parakeet_vocab.json"))) as! [String: String]
        vocab = (0..<v.count).map { v[String($0)] ?? "" }
        pieces = vocab.map { Array($0.lowercased().replacingOccurrences(of: " ", with: "▁")) }
    }

    let canBoost: Bool
    let pieces: [[Character]]        // lowercased word-pieces, word starts as "▁"
    var boost = Boost()
    var boostedSteps = 0             // decisions changed by boosting in the last transcription

    /// Partials after emitting `piece`, or nil if the piece spells no listed term from the current state.
    private func advance(_ active: Set<Partial>, _ piece: [Character]) -> Set<Partial>? {
        guard !piece.isEmpty else { return nil }
        var next = Set<Partial>()
        var starts = Array(active)
        if piece[0] == "▁" { starts += boost.keys.indices.map { Partial(term: $0, pos: 0) } }
        for p in starts {
            let key = boost.keys[p.term]
            if p.pos + piece.count <= key.count && Array(key[p.pos..<(p.pos + piece.count)]) == piece {
                next.insert(Partial(term: p.term, pos: p.pos + piece.count))
            }
        }
        return next.isEmpty ? nil : next
    }

    /// Greedy choice with boosting. Returns the chosen token (blank included).
    private func boostedChoice(_ logits: MLMultiArray, _ stock: Int32, _ active: inout Set<Partial>) -> Int32 {
        let n = vocab.count + 1
        var scores = [Float](repeating: 0, count: n)
        let stride = logits.strides[3].intValue
        logits.withUnsafeBufferPointer(ofType: Float16.self) { b in for i in 0..<n { scores[i] = Float(b[i * stride]) } }
        let top = scores.max()!
        let logZ = top + log(scores.reduce(Float(0)) { $0 + exp($1 - top) })
        let blankLP = scores[Int(blank)] - logZ
        var best = stock, bestScore = scores[Int(stock)] - logZ
        var bestNext: Set<Partial>? = nil
        if stock != blank, let next = advance(active, pieces[Int(stock)]) {   // the model's own pick spells a term too
            bestScore += boost.beta; bestNext = next
        }
        let candidates = scores.indices.filter { $0 != Int(blank) }.sorted { scores[$0] > scores[$1] }.prefix(boost.topK)
        for id in candidates {
            let lp = scores[id] - logZ
            guard lp >= boost.floor, let next = advance(active, pieces[id]) else { continue }
            let boosted = lp + boost.beta
            if boosted > bestScore && (stock != blank || lp >= blankLP - boost.blankMargin) {
                best = Int32(id); bestScore = boosted; bestNext = next
            }
        }
        if best != blank {
            active = bestNext ?? advance(active, pieces[Int(best)]) ?? []
        }
        return best
    }

    private func decoderStep(_ token: Int32, _ h: MLMultiArray, _ c: MLMultiArray) throws -> (MLMultiArray, MLMultiArray, MLMultiArray) {
        let targets = try array([1, 1], .int32); targets[0] = NSNumber(value: token)
        let length = try array([1], .int32); length[0] = 1
        let o = try dec.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "targets": targets, "target_length": length, "h_in": h, "c_in": c]))
        return (o.featureValue(for: "decoder")!.multiArrayValue!, o.featureValue(for: "h_out")!.multiArrayValue!,
                o.featureValue(for: "c_out")!.multiArrayValue!)
    }

    /// Greedy TDT decoding of one window (at most 15 s). Token times are relative to the window.
    private func window(_ samples: ArraySlice<Float>) throws -> [Token] {
        let audio = try array([1, windowSamples], .float32)
        audio.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            buf.initialize(repeating: 0)
            for (i, s) in samples.enumerated() { buf[i] = s }
        }
        let audioLength = try array([1], .int32); audioLength[0] = NSNumber(value: samples.count)
        let p = try pre.prediction(from: MLDictionaryFeatureProvider(dictionary: ["audio_signal": audio, "audio_length": audioLength]))
        let e = try enc.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "mel": p.featureValue(for: "mel")!, "mel_length": p.featureValue(for: "mel_length")!]))
        let encoded = e.featureValue(for: "encoder")!.multiArrayValue!               // [1, 1024, T]
        let frames = min(e.featureValue(for: "encoder_length")!.multiArrayValue![0].intValue, encoded.shape[2].intValue)
        let stock = try decode(encoded, frames, boosting: false)
        guard canBoost && boost.isOn else { return stock }
        // Second pass with boosting (the encoder output is reused). Only whole listed terms are taken from it, so a
        // term the audio didn't support ("Siddhart", "Povector") never reaches the text.
        return acceptTerms(stock, try decode(encoded, frames, boosting: true))
    }

    /// Words of a token list: (first token index, end index, start time).
    private func words(_ tokens: [Token]) -> [(Int, Int, Double)] {
        var out: [(Int, Int, Double)] = []
        for i in tokens.indices where i == 0 || startsWord(tokens[i]) {
            var end = i + 1
            while end < tokens.count && !startsWord(tokens[end]) { end += 1 }
            out.append((i, end, tokens[i].time))
        }
        return out
    }

    /// The stock tokens, with each boosted word that spells a whole listed term swapped in for the stock words
    /// spoken at the same time. Boosted words that aren't whole terms are ignored.
    private func acceptTerms(_ stock: [Token], _ boosted: [Token]) -> [Token] {
        let keys = Set(boost.keys.map { String($0.dropFirst()) })   // lowercased, spaces as "▁"
        var out = stock
        let bw = words(boosted)
        for (k, w) in bw.enumerated().reversed() {
            let spelled = String(boosted[w.0..<w.1].flatMap { pieces[$0.id] }).trimmingCharacters(in: CharacterSet(charactersIn: "▁"))
            let bare = spelled.trimmingCharacters(in: .punctuationCharacters)
            guard keys.contains(bare) else { continue }
            let start = w.2, end = k + 1 < bw.count ? bw[k + 1].2 : .infinity
            let sw = words(out).filter { $0.2 >= start - 0.12 && $0.2 < end - 0.01 }
            guard let first = sw.first, let last = sw.last else { continue }   // never insert a term into silence
            let replaced = String(out[first.0..<last.1].flatMap { pieces[$0.id] })
            if replaced.trimmingCharacters(in: CharacterSet(charactersIn: "▁")) == spelled { continue }
            out.replaceSubrange(first.0..<last.1, with: boosted[w.0..<w.1])
            boostedSteps += 1
        }
        return out
    }

    /// Greedy TDT decoding over encoder frames, optionally with vocabulary boosting.
    private func decode(_ encoded: MLMultiArray, _ frames: Int, boosting: Bool) throws -> [Token] {
        let s1 = encoded.strides[1].intValue, s2 = encoded.strides[2].intValue
        var (decoderOut, h, c) = try decoderStep(blank, try zeros([decoderLayers, 1, decoderWidth]), try zeros([decoderLayers, 1, decoderWidth]))
        let step = try array([1, encoderWidth, 1], .float32)
        var tokens: [Token] = []
        var t = 0, symbols = 0
        var active = Set<Partial>()
        try encoded.withUnsafeBufferPointer(ofType: Float.self) { src in
            while t < frames {
                step.withUnsafeMutableBufferPointer(ofType: Float.self) { dst, _ in
                    for k in 0..<encoderWidth { dst[k] = src[k * s1 + t * s2] }
                }
                let j = try joint.prediction(from: MLDictionaryFeatureProvider(dictionary: ["encoder_step": step, "decoder_step": decoderOut]))
                var token = j.featureValue(for: "token_id")!.multiArrayValue![0].int32Value
                if boosting {
                    token = boostedChoice(j.featureValue(for: "token_logits")!.multiArrayValue!, token, &active)
                }
                let duration = j.featureValue(for: "duration")!.multiArrayValue![0].intValue
                if token != blank {
                    tokens.append(Token(id: Int(token), time: Double(t) * secondsPerFrame))
                    (decoderOut, h, c) = try decoderStep(token, h, c)
                }
                t += duration
                symbols += 1
                if duration != 0 { symbols = 0 } else if symbols >= maxSymbolsPerFrame { t += 1; symbols = 0 }
            }
        }
        return tokens
    }

    // Word starts are marked with a leading space in this vocabulary ("▁" in SentencePiece files).
    private func startsWord(_ t: Token) -> Bool { vocab[t.id].hasPrefix(" ") || vocab[t.id].hasPrefix("▁") }

    /// The word that starts at `tokens[i]`, lowercased and without punctuation.
    private func word(_ tokens: [Token], at i: Int) -> String {
        var end = i + 1
        while end < tokens.count && !startsWord(tokens[end]) { end += 1 }
        return tokens[i..<end].map { vocab[$0.id] }.joined().lowercased().filter { $0.isLetter || $0.isNumber }
    }

    func transcribe(_ samples: [Float]) throws -> String {
        guard samples.count >= 160 else { return "" }
        var all: [Token] = []
        var start = 0
        while true {
            let end = min(start + windowSamples, samples.count)
            let offset = Double(start) / Double(sampleRate)
            var tokens = try window(samples[start..<end]).map { Token(id: $0.id, time: $0.time + offset) }
            if start > 0 {
                // Overlap with the previous window is [offset, offset + 3 s). Cut both at the first word
                // starting at or after its midpoint, so each word comes from exactly one window.
                let midpoint = offset + Double(windowSamples - strideSamples) / Double(sampleRate) / 2
                var cutTime = midpoint
                var cutWord: String?
                if let cut = all.firstIndex(where: { startsWord($0) && $0.time >= midpoint }) {
                    cutTime = all[cut].time
                    cutWord = word(all, at: cut)
                    all.removeSubrange(cut...)
                }
                // Timestamps for the same spoken word differ slightly between windows. Resume at the same
                // word near the cut; otherwise at the nearest word start; otherwise the first after it.
                let near = tokens.indices.filter { startsWord(tokens[$0]) && abs(tokens[$0].time - cutTime) <= 0.4 }
                    .sorted { abs(tokens[$0].time - cutTime) < abs(tokens[$1].time - cutTime) }
                let from = near.first { word(tokens, at: $0) == cutWord } ?? near.first
                    ?? tokens.indices.first { startsWord(tokens[$0]) && tokens[$0].time >= cutTime } ?? tokens.count
                tokens = Array(tokens[from...])
            }
            all += tokens
            if end == samples.count { break }
            start += strideSamples
        }
        let text = all.map { vocab[$0.id] }.joined().replacingOccurrences(of: "▁", with: " ")
            .trimmingCharacters(in: .whitespaces)
        return boost.isOn && canBoost ? spellTerms(text) : text
    }

    /// Use the listed spelling for any whole word that matches a boosted term ignoring case ("grafana" -> "Grafana").
    private func spellTerms(_ text: String) -> String {
        var out = text
        for term in boost.terms.sorted(by: { $0.count > $1.count }) {
            let pattern = "(?<![\\w-])" + NSRegularExpression.escapedPattern(for: term) + "(?![\\w-])"
            guard let re = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { continue }
            out = re.stringByReplacingMatches(in: out, range: NSRange(out.startIndex..., in: out),
                                              withTemplate: NSRegularExpression.escapedTemplate(for: term))
        }
        return out
    }
}

let args = CommandLine.arguments
guard args.count >= 3 else {
    FileHandle.standardError.write(Data("usage: orukeet-ane <bundleDir> <cacheDir> [--warmup]\n".utf8))
    exit(2)
}
let started = Date()
let model: Orukeet
do {
    model = try Orukeet(bundle: URL(fileURLWithPath: args[1]), cache: URL(fileURLWithPath: args[2]))
    _ = try model.transcribe([Float](repeating: 0, count: sampleRate))   // build Neural Engine kernels now
} catch {
    emit(["error": "load failed: \(error)"])
    exit(1)
}
// "boost": this helper understands JSON request lines and can boost (the server checks before sending terms).
emit(["ready": true, "load_ms": Int(Date().timeIntervalSince(started) * 1000), "boost": model.canBoost])
if args.contains("--warmup") { exit(0) }

// Each stdin line is a WAV path, or a JSON object {"path": ..., "boost": [terms], "beta": ..., ...}.
// Terms are used for this request only and never logged.
while let line = readLine() {
    let raw = line.trimmingCharacters(in: .whitespacesAndNewlines)
    if raw.isEmpty { continue }
    var path = raw
    model.boost = Boost()
    if raw.hasPrefix("{") {
        guard let obj = try? JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any],
              let p = obj["path"] as? String else { emit(["error": "bad request line"]); continue }
        path = p
        model.boost = Boost(terms: obj["boost"] as? [String] ?? [], options: obj)
    }
    model.boostedSteps = 0
    let t = Date()
    do {
        let text = try model.transcribe(try readWav(URL(fileURLWithPath: path)))
        emit(["text": text, "ms": Int(Date().timeIntervalSince(t) * 1000), "boosted": model.boostedSteps])
    } catch {
        emit(["error": "\(error)"])
    }
}
