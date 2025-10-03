# Neptune Exporter Project - Task Decomposition

## Overview
Migration tool to help Neptune customers transition their data out of neptune in case of acquisition.

## Migration Paths

**Source:**
- Neptune 3.x (using neptune-query)
- Neptune 2.x (using neptune-client)

**Target:**
- parquet files
- MLflow (optional?)
- W&B (optional?)

## Requirements

### Data Volume & Performance
- **Large datasets**: May need chunking, batching, concurrency, rate limiting - neptune/other clients have most of these built in. Concurrency is worth considering
- **File artifacts**: Should be exported as well. Will likely need to be handled as separate objects from the metrics. Streaming for large files
- **Memory usage**: Efficient data processing, avoid fully buffering data

### Data Filtering Options
- **Experiment filtering**: by name, nql
- **Attribute filtering**: by name, type
Neptune clients have filtering capabilities. They should be exposed.

### Migration Process
1. **Discovery**: Scan Neptune project structure
2. **Selection**: Choose experiments/runs to migrate
3. **Extraction**: Download data from Neptune
4. **Transformation**: Convert to target format
5. **Loading**: Upload to target platform
6. **Validation**: Verify data integrity and basic functionality

### Durability & Resumability
- **Durability**: Data is stored on disk, either as a final or intermediate step
- **Resumable migrations**: Resume after interruptions - discover existing files and avoid exporting experiments that are already complete
- **Progress tracking**: Real-time progress bars and status updates
- **Error handling**: Detailed error reporting

### Validation
- **Data completeness**: 100% of selected data migrated
- **Data accuracy**: All metrics, parameters, artifacts match
- **Data functionality**: Data is accessible and usable in target platform
- It'd probably be just nice to have.

## Implementation Architecture

### Core Components
```
neptune_exporter/
├── main.py                       # Cli entry point
├── model.py                      # Data model used in between the steps
├── exporters/
│   ├── neptune3.py               # Using neptune-query
│   └── neptune2.py               # Using neptune-client
├── storage/
│   └── parquet.py                # Using pyarrow
├── loaders/
│   ├── mlflow.py
│   └── wandb.py
```

## Implementation Plan

### Core Functionality
1. **Project Setup**

2. **Basic Export (Neptune 3.x → Parquet)**
   - Implement neptune3.py exporter
   - Implement parquet storage, design Parquet schema
   - Create basic CLI for export
   - Create basic export manager orchestrating the export using exporter and storage classes

3. **Artifact Export (Neptune 3.x)**
   - Implement download of files/artifacts

### Multi-Source and Target Support
4. **Neptune 2.x Export**

5. **W&B Loader**
   - Implement wandb.py loader
   - Handle data type conversions (see model.md)
   - Test with real W&B projects (could be a challenge as we are mostly limited to SaaS)

6. **MLflow Loader**
   - Implement mlflow.py loader
   - Handle data type conversions (see model.md)
   - Test with real MLflow instances

### Advanced Features
7. **Filtering & Selection**
   - Extend the APIs and CLI to support project, experiment and attribute filtering

8. **Progress & Resumability**
   - Add progress tracking
   - Add resumability features (do not repeat the export if the parquet/artifact files are already present)
   - Error handling with detailed messages

### Quality and Documentation
10. **Documentation & Examples**
   - Add example usage
   - Create user documentation
   - Help and better UX in CLI

11. **Validation**
   - Validate data completeness and accuracy after the export
