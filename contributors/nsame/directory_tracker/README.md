# Understanding Git Internals

Git is a version control system that stores code change a developer makes as a commit. Each commit has a unique ID, a timestamp, a message, and list of file changes (the diff).

`git log` is used to read this history programmatically. 

Git keeps track of everything the developer did.

Flags and their uses:
- "--stat": display the number of insertions and deletions for each file
- "--p": Shows full patch/diff of changes for each commit
- "--graph": Draws an ASCII-based textual representation of the branch
- "--all": Shows history for all branches
- "-n": limits the number of recent commits
- "--author='<name>'": shows only commits made by a specific author
- "--since='<date>' / --until='<date>'"
- there's more but for project zila, this is satisfactory.

## The Core Data Objects
Git manages version control using three major immutable objects stord in the .git/objects directory.
1. **Blobs**: These store the file content of the files. In Git, it is the content itself that is tracked rather than the file. If two files only have identical contetn, Git stores that blob once.

2. **Trees**: These correspond to the directories. A tree lists the filenames and modes, pointing to the SHA-1 hashes of blobs or other subdirectories.

3. **Commits**: These objects store the project history pointing to a specific top-level tree snapshot.

**Snapshot Instead of Deltas**
Unlike many other version control systems that store changes as a series of deltas or diffs, Git stores full snapshots of the project. When a new version is committed, Git records a new tree; if a file has not changed, ***Git simply reuses the reference to the existing blob in the database.** This approach makes operations like branching and merging highly efficient because Git only needs to compare top-level trees rather than recalculating a chain of deltas

**The role of the Index**
Git utilizes a middle ground between working directory and the repository known as the index or staging area. 

When you run git add, changes are moved from the working directory into the index. The index acts as a staging area where you can carefully curate exactly which changes will be included in the next commit. 

When git commit is executed, Git builds the new tree and commit object based on the current state of this index

**Branching and merging**
Branching in Git is uniquely "cheap" because a branch is nothing more than a moveable pointer (a reference) to a specific commit SHA-1.
Creating a new branch involves simply writing 40 characters to a file in the .git/refs/heads directory.

Merging occurs when these lines of development are combined, often resulting in a merge commit that points to multiple parent commits.

**Distributed Architecture**
Git is a distributed system, meaning every clone of a repository is a complete backup containing the full history of every file and every version. This architecture allows for various workflows:
- **Local Operations**: Most operations are local and do not require network overhead, making the system very fast.
- **Remotes**: Developers can collaborate by fetching (downloading objects and references) or pushing (uploading objects and advancing remote branches) changes between different copies of the repository.
- **Decentralized Collaboration**: Because no single repository is "special," developers can share changes through multiple remotes, such as a "blessed" central repository or individual public forks
