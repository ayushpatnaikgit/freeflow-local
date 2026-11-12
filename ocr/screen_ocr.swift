// screen-ocr: read the text in FreeFlow's context screenshot with Apple's Vision framework, on device.
//
//   screen-ocr            serve: one line of base64 image data (or a data: URL) per stdin line -> one JSON line
//   screen-ocr --selftest render a synthetic window, read it back, exit 0 if the planted words were found
//
// FreeFlow sends the active window as a JPEG scaled to 1024 px on its longest side, so editor text is only
// 8-9 px tall. Vision's accurate recogniser misses much of that, and on dark themes it often returns nothing
// at all. Reading the image upscaled 2x, both as is and with inverted colours, recovers most of it
// (synthetic dark/light, sans/mono, 11-14 pt: 146/160 planted words vs 96/160 for a single plain pass).
// Nothing is written to disk; the image and text only pass through this process's memory.
import AppKit
import CoreImage
import Vision

let ci = CIContext(options: [.cacheIntermediates: false])

func emit(_ object: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: object)
    FileHandle.standardOutput.write(data + Data("\n".utf8))
}

func decode(_ line: String) -> CGImage? {
    var b64 = Substring(line)
    if b64.hasPrefix("data:"), let comma = b64.firstIndex(of: ",") { b64 = b64[b64.index(after: comma)...] }
    guard let data = Data(base64Encoded: String(b64), options: .ignoreUnknownCharacters),
          let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(src, 0, nil)
}

func render(_ image: CIImage) -> CGImage? { ci.createCGImage(image, from: image.extent) }

func recognise(_ image: CGImage) -> [String] {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false  // keep LoRA, pgvector and names exactly as drawn
    do { try VNImageRequestHandler(cgImage: image).perform([request]) } catch { return [] }
    return (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }
}

/// Overlapping tiles (25% overlap) covering the image in `rows` x `cols`.
func tiles(_ image: CGImage, rows: Int, cols: Int) -> [CGImage] {
    let tw = Double(image.width) / (Double(cols) * 0.75 + 0.25)
    let th = Double(image.height) / (Double(rows) * 0.75 + 0.25)
    var out: [CGImage] = []
    for r in 0..<rows {
        for c in 0..<cols {
            let rect = CGRect(x: Double(c) * tw * 0.75, y: Double(r) * th * 0.75, width: tw, height: th).integral
            if let tile = image.cropping(to: rect) { out.append(tile) }
        }
    }
    return out
}

/// Text lines from every tile, as drawn and inverted (a line read in several passes appears several times).
///
/// Whole-image passes are unreliable at this size: on a window with one short line of 14 pt text, Vision found
/// nothing in the 1x, 2x, plain or inverted image, but read it in 3x2 tiles of the 2x image. On synthetic dark
/// and light, sans and mono, 11-14 pt, dense and sparse windows: 305/320 planted words, worst case 8/10.
func read(_ image: CGImage) -> [String] {
    var source = CIImage(cgImage: image)
    if max(image.width, image.height) <= 2048 {
        source = source.applyingFilter("CILanczosScaleTransform", parameters: ["inputScale": 2.0, "inputAspectRatio": 1.0])
    }
    guard let big = render(source) else { return [] }
    var passes: [CGImage] = []
    for tile in tiles(big, rows: 3, cols: 2) {
        passes.append(tile)
        if let inverted = render(CIImage(cgImage: tile).applyingFilter("CIColorInvert")) { passes.append(inverted) }
    }
    // Sequential: running the passes concurrently was slower (~700 vs ~500 ms on an M4). Every pass's lines are
    // returned, repeats included: a word read the same way in several passes is real, OCR misreads vary.
    return passes.flatMap(recognise)
}

func selftest() -> Bool {
    let words = ["LoRA", "Orukeet", "Siddhant", "Grafana"]  // "pgvector" reads as "pavector" at this size
    let text = "Train a LoRA adapter, then ask Siddhant about Orukeet, Grafana and pgvector."
    let (w, h) = (3024, 1890)  // a Retina window, scaled down like FreeFlow does below
    guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { return false }
    ctx.setFillColor(CGColor(red: 0.12, green: 0.12, blue: 0.13, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: false)
    NSAttributedString(string: text, attributes: [.font: NSFont.systemFont(ofSize: 28),
                                                  .foregroundColor: NSColor(white: 0.85, alpha: 1)])
        .draw(at: CGPoint(x: 80, y: h - 400))
    NSGraphicsContext.restoreGraphicsState()
    // Same as FreeFlow: longest side to 1024 px, JPEG at 0.5, sent as a base64 data URL.
    let (sw, sh) = (1024, 1024 * h / w)
    guard let full = ctx.makeImage(),
          let small = CGContext(data: nil, width: sw, height: sh, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { return false }
    small.interpolationQuality = .high
    small.draw(full, in: CGRect(x: 0, y: 0, width: sw, height: sh))
    guard let scaled = small.makeImage(),
          let jpeg = NSBitmapImageRep(cgImage: scaled).representation(using: .jpeg, properties: [.compressionFactor: 0.5]),
          let image = decode("data:image/jpeg;base64," + jpeg.base64EncodedString())
    else { return false }
    let found = read(image).joined(separator: " ")
    let missing = words.filter { !found.contains($0) }
    emit(["selftest": missing.isEmpty, "missing": missing])
    return missing.isEmpty
}

if CommandLine.arguments.contains("--selftest") { exit(selftest() ? 0 : 1) }

emit(["ready": true])
while let line = readLine(strippingNewline: true) {
    if line.isEmpty { continue }
    let started = Date()
    guard let image = decode(line) else {
        emit(["error": "could not decode image"])
        continue
    }
    let lines = autoreleasepool { read(image) }
    emit(["lines": lines, "ms": Int(Date().timeIntervalSince(started) * 1000)])
}
