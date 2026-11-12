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
        joint = try model("JointDecisionv3")
        let v = try JSONSerialization.jsonObject(with: Data(contentsOf: bundle.appendingPathComponent("parakeet_vocab.json"))) as! [String: String]
        vocab = (0..<v.count).map { v[String($0)] ?? "" }
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
        let s1 = encoded.strides[1].intValue, s2 = encoded.strides[2].intValue

        var (decoderOut, h, c) = try decoderStep(blank, try zeros([decoderLayers, 1, decoderWidth]), try zeros([decoderLayers, 1, decoderWidth]))
        let step = try array([1, encoderWidth, 1], .float32)
        var tokens: [Token] = []
        var t = 0, symbols = 0
        try encoded.withUnsafeBufferPointer(ofType: Float.self) { src in
            while t < frames {
                step.withUnsafeMutableBufferPointer(ofType: Float.self) { dst, _ in
                    for k in 0..<encoderWidth { dst[k] = src[k * s1 + t * s2] }
                }
                let j = try joint.prediction(from: MLDictionaryFeatureProvider(dictionary: ["encoder_step": step, "decoder_step": decoderOut]))
                let token = j.featureValue(for: "token_id")!.multiArrayValue![0].int32Value
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
        return all.map { vocab[$0.id] }.joined().replacingOccurrences(of: "▁", with: " ")
            .trimmingCharacters(in: .whitespaces)
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
emit(["ready": true, "load_ms": Int(Date().timeIntervalSince(started) * 1000)])
if args.contains("--warmup") { exit(0) }

while let line = readLine() {
    let path = line.trimmingCharacters(in: .whitespacesAndNewlines)
    if path.isEmpty { continue }
    let t = Date()
    do {
        let text = try model.transcribe(try readWav(URL(fileURLWithPath: path)))
        emit(["text": text, "ms": Int(Date().timeIntervalSince(t) * 1000)])
    } catch {
        emit(["error": "\(error)"])
    }
}
