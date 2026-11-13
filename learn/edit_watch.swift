// edit-watch: after FreeFlow pastes a cleaned transcript, watch the focused text field for the user's corrections.
//
// Used only when vocabulary learning is on (freeflowctl learn on). Needs Accessibility permission for this binary.
// The process relaunches itself "disclaimed" so macOS attributes that permission to edit-watch itself, not to the
// Python server that started it.
//
// Protocol: one JSON object per stdin line, one JSON line out per finished watch.
//   {"watch": "<pasted text>", "seconds": 30}   start watching (finishes any previous watch first)
//   {"stop": true}                               finish the current watch now (a new dictation is starting)
//   out: {"edited": "<the pasted stretch as it reads now>"} or {"edited": null} (unchanged, unreadable, or lost)
// Text stays in memory: nothing is written to disk or logged here. Password fields are never read.
//
//   edit-watch --request-permission   show macOS's Accessibility prompt for this binary, then exit
import ApplicationServices
import Darwin
import Foundation

// ------------------------------------------------------------------ run as our own responsible process
typealias Disclaim = @convention(c) (UnsafeMutablePointer<posix_spawnattr_t?>, Int32) -> Int32

func relaunchDisclaimed() -> Never {
    var attr: posix_spawnattr_t? = nil
    posix_spawnattr_init(&attr)
    if let f = dlsym(dlopen(nil, RTLD_NOW), "responsibility_spawnattrs_setdisclaim") {
        _ = unsafeBitCast(f, to: Disclaim.self)(&attr, 1)
    }
    let path = CommandLine.arguments[0]
    var argv: [UnsafeMutablePointer<CChar>?] = [strdup(path), strdup("--child")]
    argv += CommandLine.arguments.dropFirst().map { strdup($0) }
    argv.append(nil)
    var pid: pid_t = 0
    guard posix_spawn(&pid, path, nil, &attr, argv, environ) == 0 else { exit(1) }
    signal(SIGTERM) { _ in exit(0) }   // the child notices stdin closing and exits too
    var status: Int32 = 0
    waitpid(pid, &status, 0)
    exit((status >> 8) & 0xff)
}

if !CommandLine.arguments.contains("--child") { relaunchDisclaimed() }

if CommandLine.arguments.contains("--request-permission") {
    let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
    print(AXIsProcessTrustedWithOptions([key: true] as CFDictionary) ? "already allowed" : "asked")
    exit(0)
}

// ------------------------------------------------------------------ output
let outLock = NSLock()
func emit(_ object: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: object)
    outLock.lock(); FileHandle.standardOutput.write(data + Data("\n".utf8)); outLock.unlock()
}

// ------------------------------------------------------------------ Accessibility reads
let systemWide = AXUIElementCreateSystemWide()

func attribute(_ element: AXUIElement, _ name: String) -> AnyObject? {
    var value: AnyObject?
    return AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success ? value : nil
}

/// The focused text element, unless it is a password field.
func focusedField() -> AXUIElement? {
    guard let raw = attribute(systemWide, kAXFocusedUIElementAttribute as String),
          CFGetTypeID(raw) == AXUIElementGetTypeID() else { return nil }
    let element = raw as! AXUIElement
    if let subrole = attribute(element, kAXSubroleAttribute as String) as? String,
       subrole == (kAXSecureTextFieldSubrole as String) { return nil }
    return element
}

func text(of element: AXUIElement) -> String? { attribute(element, kAXValueAttribute as String) as? String }

// ------------------------------------------------------------------ watching
let anchorLength = 40
var generation = 0
var stopRequested = 0          // generation that was asked to stop
let state = NSLock()
let queue = DispatchQueue(label: "watch")

func watch(_ pasted: String, seconds: Double, gen: Int) {
    let target = pasted.trimmingCharacters(in: .whitespacesAndNewlines)
    func current() -> Bool { state.lock(); defer { state.unlock() }; return generation == gen && stopRequested != gen }
    guard !target.isEmpty else { emit(["edited": NSNull()]); return }

    // Find the field and the pasted text in it (FreeFlow pastes right after the cleanup reply).
    Thread.sleep(forTimeInterval: 0.6)
    var field: AXUIElement?
    var prefix = "", suffix = ""
    for _ in 0..<8 where current() {
        if let f = focusedField(), let value = text(of: f), let r = value.range(of: target, options: .backwards) {
            field = f
            prefix = String(value[..<r.lowerBound].suffix(anchorLength))
            suffix = String(value[r.upperBound...].prefix(anchorLength))
            break
        }
        Thread.sleep(forTimeInterval: 0.3)
    }
    guard let field else { emit(["edited": NSNull()]); return }

    // Poll until the time is up, focus moves to another field, or we're told to stop.
    var last = text(of: field) ?? ""
    let deadline = Date().addingTimeInterval(seconds)
    while Date() < deadline && current() {
        Thread.sleep(forTimeInterval: 0.5)
        guard let now = focusedField(), CFEqual(now, field) else { break }
        if let value = text(of: field) { last = value }
    }

    // The pasted stretch as it reads now: between the text that surrounded it.
    var start = last.startIndex, end = last.endIndex
    if !prefix.isEmpty {
        guard let p = last.range(of: prefix) else { emit(["edited": NSNull()]); return }
        start = p.upperBound
    }
    if !suffix.isEmpty {
        guard let s = last.range(of: suffix, range: start..<last.endIndex) else { emit(["edited": NSNull()]); return }
        end = s.lowerBound
    }
    let edited = String(last[start..<end]).trimmingCharacters(in: .whitespacesAndNewlines)
    emit(["edited": edited == target || edited.isEmpty ? NSNull() : edited as Any])
}

emit(["ready": true, "trusted": AXIsProcessTrusted()])

while let line = readLine() {
    guard let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any] else { continue }
    if obj["stop"] != nil {
        state.lock(); stopRequested = generation; state.unlock()
    } else if let pasted = obj["watch"] as? String {
        state.lock(); stopRequested = generation; generation += 1; let gen = generation; state.unlock()
        let seconds = (obj["seconds"] as? Double) ?? 30
        queue.async { watch(pasted, seconds: seconds, gen: gen) }   // serial: a previous watch finishes first
    }
}
