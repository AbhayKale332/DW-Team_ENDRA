Yes. The UI is becoming much cleaner. I would structure it like this:

### Top Header

```text
DepthWizard    File ▾    View ▾    Tools ▾    Help ▾
                                     
                              [3D DSM View ▾]
```

The **3D DSM View dropdown** is a quick-access control directly over the viewport:

```text
3D DSM View ▾
├── 3D Height Map (DSM)
├── Height Map
└── Input Image
```

This is the fastest way to switch the primary visualization.

### `File ▾`

```text
File
├── New Project
├── Load / Open
├── Recent
├── Save Project
└── Export
    ├── 3D Object
    │   ├── GLB
    │   ├── OBJ
    │   └── other supported scientific formats
    ├── Heatmap
        ├── PNG
        └── JPG
```

### `View ▾`

```text
View
├── Input Image
├── Heatmap
└── 3D Height Map (DSM)
```

This is essentially the full visualization/navigation menu, while the small dropdown above the viewport is the **quick version**.

### `Tools ▾`

```text
Tools
├── Camera Control
└── Flight Simulator
```

Later you can add more plugins without changing the main architecture.

### `Help ▾`

```text
Help
├── Documentation
├── Keyboard Shortcuts
├── Model Information
└── About DepthWizard
```

### Always visible on the 3D viewport

Keep only the controls that are genuinely useful during interaction:

**Compass:**
`N / E / S / W`

**Right-side controls:**
` + | − | Fullscreen | reset [only when zoom is enabled]`

